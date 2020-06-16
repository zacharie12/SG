# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import argparse
import datetime
import os
import time
from copy import deepcopy

import PIL
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
import wandb
from PIL import Image
from pytorch_memlab import profile
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms

# Speaker Listener imports
import maskrcnn_benchmark.listener
import maskrcnn_benchmark.structures.image_list
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import listener_inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.listener.listener import build_listener
from maskrcnn_benchmark.listener.utils import collate_sgs, format_scores
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import (make_listener_optimizer,
                                       make_lr_scheduler, make_optimizer)
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import (Checkpointer,
                                                 DetectronCheckpointer,
                                                 clip_grad_norm,
                                                 clip_grad_value)
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import (all_gather, get_rank,
                                           is_main_process, synchronize)
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import debug_print, setup_logger
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

@profile
def train(cfg, local_rank, distributed, logger):
    if is_main_process():
        wandb.init(project='scene-graph', entity='sgg-speaker-listener')
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg)
    listener = build_listener(cfg)
    if is_main_process():
        wandb.watch(listener)
        

    debug_print(logger, 'end model construction')

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
 
    fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}
    
    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    listener.to(device)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
    listener_optimizer = make_listener_optimizer(cfg, listener)
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    listener_scheduler = None
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    listener, listener_optimizer = amp.initialize(listener, listener_optimizer, opt_level='O0')
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    listener.float()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        listener = torch.nn.parallel.DistributedDataParallel(
            listener, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        ) 
    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR
    listener_dir = cfg.LISTENER_DIR 
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    listener_checkpointer = Checkpointer(
        listener, listener_optimizer, listener_scheduler, listener_dir, save_to_disk, custom_scheduler=False
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if listener_checkpointer.has_checkpoint():
        extra_listener_checkpoint_data = listener_checkpointer.load('')
        arguments.update(extra_listener_checkpoint_data)

    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, 
                                       update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
    debug_print(logger, 'end load checkpointer')
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        ret_images=True
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
        ret_images=True
    )
    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        #loss_val =  run_val(cfg, model, listener, val_data_loaders, distributed, logger)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    print_first_grad = True

    listener_loss_func = torch.nn.MarginRankingLoss(margin=0.2, reduction='none')
    while True:
        try:
            for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
                print(f'ITERATION NUMBER: {iteration}')
                if any(len(target) < 1 for target in targets):
                    logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
                data_time = time.time() - end
                iteration = iteration + 1
                arguments["iteration"] = iteration

                model.train()
                fix_eval_modules(eval_modules)

                images_list = deepcopy(images)
                images_list = to_image_list(images_list, cfg.DATALOADER.SIZE_DIVISIBILITY).to(device)

                '''
                SAVE IMAGE TO PC
                is_printed = False
                if is_main_process():
                    if not is_printed:
                        transform = transforms.ToPILImage()
                        print('SAVING IMAGE')
                        img = transform(images[0])
                        print('DONE TRANSFORM')
                        img.save('img.png')
                        print('DONE SAVING IMAGE')
                        is_printed = True
                '''
                '''
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    #transforms.Resize((cfg.LISTENER.IMAGE_SIZE, cfg.LISTENER.IMAGE_SIZE)),
                    transforms.ToTensor(),
                ])
                '''
                # turn images to a uniform size
                #print('IMAGE BEFORE Transform: ', images[0], 'GPU: ', get_rank())
                for i in range(len(images)):
                    images[i] = images[i].unsqueeze(0)
                    images[i] = F.interpolate(images[i], size=(224, 224), mode='bilinear', align_corners=False)
                    images[i] = images[i].squeeze()

                images = torch.stack(images).to(device)
                #images.requires_grad_()

                targets = [target.to(device) for target in targets]

                #print('IMAGE BEFORE Model: ', images[0], 'GPU: ', get_rank())

                _, sgs = model(images_list, targets)

                #print('IMAGE AFTER Model: ', images)
                '''is_printed = False
                if is_main_process():
                    if not is_printed:
                        print('PRINTING OBJECTS')
                        (obj, rel_pair, rel) = sgs[0]
                        obj = torch.argmax(obj, dim=1)
                        for i in range(obj.size(0)):
                            print(f'OBJECT {i}: ', obj[i])
                        print('DONE PRINTING OBJECTS')
                        is_printed=True'''

                
                image_list = None
                sgs = collate_sgs(sgs, cfg.MODEL.DEVICE)

                listener_loss = None
                gap_reward = 0
                avg_acc = 0
                num_correct = 0
                for true_index, sg in enumerate(sgs):
                    acc = 0
                    detached_sg = (sg[0].detach().requires_grad_(), sg[1], sg[2].detach().requires_grad_() )
                    #scores = listener(sg, images)
                    scores = listener(detached_sg, images)
                    print('SCORES: ', scores, ' \nRight index: ', true_index)
                    (predicted_scores, true_scores, _) = format_scores(scores, true_index, device)
                    predicted_scores = predicted_scores.t()
                    true_scores = true_scores.t()
                    is_correct = True
                    for i in range(len(predicted_scores)):
                        if predicted_scores[i] < true_scores[i]:
                            acc += 1
                        else:
                            is_correct = False
                    
                    acc  = 100*acc / (len(predicted_scores) - 1)

                    (true_tensor, scores, binary) = format_scores(scores, true_index, device)

                    if listener_loss is None:
                        listener_loss = torch.max(listener_loss_func(true_tensor, scores, binary))
                    else:
                        listener_loss = listener_loss + torch.max(listener_loss_func(true_tensor, scores, binary))


                    # add a loss term that aims at maximizing the gap between scores, so the model
                    # won't get stuck in a local minima where every pair gets the same score
                    if is_correct:
                        num_correct += 1
                        if gap_reward == 0:
                            gap_reward = -torch.sum(torch.abs(scores - true_tensor)) * cfg.LISTENER.GAP_COEF / (len(predicted_scores) - 1)
                        else:
                            gap_reward = gap_reward -torch.sum(torch.abs(scores - true_tensor)) * cfg.LISTENER.GAP_COEF / (len(predicted_scores) - 1)                    



                    avg_acc += acc

                avg_acc /= len(sgs)

                avg_acc = torch.tensor([avg_acc]).to(device)
                # reduce acc over all gpus
                avg_acc = {'acc' : avg_acc}
                avg_acc_reduced = reduce_loss_dict(avg_acc)
                avg_acc_reduced = sum(acc for acc in avg_acc_reduced.values())
                # log acc to wadb
                if is_main_process():
                    wandb.log({"Train Accuracy": avg_acc_reduced.item()})

                num_sgs = true_index + 1
                listener_loss /= num_sgs
                if num_correct != 0:
                    gap_reward /= num_correct
                listener_loss *= cfg.LISTENER.LOSS_COEF
                listener_loss += gap_reward
                
                listener_loss = listener_loss.to(device)
                print('LISTENER_LOSS: ', listener_loss)
                loss_dict = {
                    'LISTENER_LOSS' : listener_loss
                }

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if is_main_process():
                    wandb.log({"Train Loss": losses_reduced})
                meters.update(loss=losses_reduced, **loss_dict_reduced)

                listener_optimizer.zero_grad()
                # Note: If mixed precision is not used, this ends up doing nothing
                # Otherwise apply loss scaling for mixed-precision recipe
                with amp.scale_loss(losses, listener_optimizer) as scaled_losses:
                    scaled_losses.backward()
                
                verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
                print_first_grad = False
                #clip_grad_value([(n, p) for n, p in listener.named_parameters() if p.requires_grad], cfg.LISTENER.CLIP_VALUE, logger=logger, verbose=True, clip=True)
                listener_optimizer.step()

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)

                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if iteration % 200 == 0 or iteration == max_iter:
                    logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=listener_optimizer.param_groups[-1]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )

                if iteration % checkpoint_period == 0:
                    listener_checkpointer.save("model_{:07d}".format(iteration), **arguments)
                if iteration == max_iter:
                    listener_checkpointer.save("model_final", **arguments)

                val_result = None # used for scheduler updating
                if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
                    logger.info("Start validating")
                    val_result = run_val(cfg, model, listener, val_data_loaders, distributed, logger)
                    if is_main_process():
                        wandb.log({"Validation Loss": val_result})
                    logger.info("Validation Result: %.4f" % val_result)
        except:
            train_data_loader = make_data_loader(
                cfg,
                mode='train',
                is_distributed=distributed,
                start_iter=arguments["iteration"],
                ret_images=True
            )
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return listener

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def run_val(cfg, model, listener, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = listener_inference(
                            cfg,
                            model,
                            listener,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=None,
                            logger=logger,
                        )
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    print('VAL_RESULT: ', val_result)
    gathered_result = all_gather(torch.tensor(val_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    torch.cuda.empty_cache()
    return val_result

def run_test(cfg, model, listener, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        listener_inference(
            cfg,
            model,
            listener,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.NUM_GPUS = num_gpus
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    listener_dir = cfg.LISTENER_DIR
    if listener_dir:
        mkdir(listener_dir)    

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    """
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    """


    listener_config_path = os.path.join(cfg.LISTENER_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(listener_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, listener_config_path)

    listener = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test:
        run_test(cfg, model, listener, args.distributed, logger)


if __name__ == "__main__":
    main()
