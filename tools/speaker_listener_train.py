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
import GPUtil
import PIL
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
import wandb
from PIL import Image
from pytorch_memlab import profile, MemReporter
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
# Speaker Listener imports
import maskrcnn_benchmark.listener
import maskrcnn_benchmark.structures.image_list
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import listener_inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.listener.listener import build_listener
from maskrcnn_benchmark.listener.utils import collate_sgs, format_scores, format_scores_reg, MistakeSaver, load_vg_info
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
        wandb.init(project='scene-graph', entity='sgg-speaker-listener', config=cfg.LISTENER)
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
        listener, optimizer=listener_optimizer, save_dir=listener_dir, save_to_disk=save_to_disk, custom_scheduler=False
    )

    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if listener_checkpointer.has_checkpoint():
        extra_listener_checkpoint_data = listener_checkpointer.load()
        listener = listener.to(torch.float32)

        '''
        print('Weights after load: ')
        print('****************************')
        print(listener.gnn.conv1.node_model.node_mlp_1[0].weight)
        print('****************************')
        '''
        # arguments.update(extra_listener_checkpoint_data)

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
        #output =  run_val(cfg, model, listener, val_data_loaders, distributed, logger)
        #print('OUTPUT: ', output)
        #(sg_loss, img_loss, sg_acc, img_acc) = output

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    print_first_grad = True

    listener_loss_func = torch.nn.MarginRankingLoss(margin=1, reduction='none')
    mistake_saver = None
    if is_main_process():
        ds_catalog = DatasetCatalog()
        dict_file_path = os.path.join(ds_catalog.DATA_DIR, ds_catalog.DATASETS['VG_stanford_filtered_with_attribute']['dict_file'])
        ind_to_classes, ind_to_predicates = load_vg_info(dict_file_path)
        ind_to_classes = {k:v for k,v in enumerate(ind_to_classes)}
        ind_to_predicates = {k:v for k,v in enumerate(ind_to_predicates)}
        print('ind to classes:', ind_to_classes, '/n ind to predicates:', ind_to_predicates)
        mistake_saver = MistakeSaver('/Scene-Graph-Benchmark.pytorch/filenames_masked', ind_to_classes, ind_to_predicates)

    #is_printed = False
    while True:
        try:
            listener_iteration=0
            for iteration, (images, targets, image_ids) in enumerate(train_data_loader, start_iter):
                listener_optimizer.zero_grad()

                #print(f'ITERATION NUMBER: {iteration}')
                if any(len(target) < 1 for target in targets):
                    logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
                if len(images) <= 1:
                    continue

                data_time = time.time() - end
                iteration = iteration + 1
                listener_iteration += 1
                arguments["iteration"] = iteration
                model.train()
                fix_eval_modules(eval_modules)
                images_list = deepcopy(images)
                images_list = to_image_list(images_list, cfg.DATALOADER.SIZE_DIVISIBILITY).to(device)
               
                #SAVE IMAGE TO PC

                
                
                '''
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    #transforms.Resize((cfg.LISTENER.IMAGE_SIZE, cfg.LISTENER.IMAGE_SIZE)),
                    transforms.ToTensor(),
                ])
                '''
                # turn images to a uniform size
                #print('IMAGE BEFORE Transform: ', images[0], 'GPU: ', get_rank())
                '''

                if is_main_process():
                    if not is_printed:
                        transform = transforms.ToPILImage()
                        print('SAVING IMAGE')
                        img = transform(images[0].cpu())
                        print('DONE TRANSFORM')
                        img.save('image.png')
                        print('DONE SAVING IMAGE')
                        print('ids ', image_ids[0])

                '''        

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
                '''
                is_printed = False
                if is_main_process():
                    if not is_printed:
                        print('PRINTING OBJECTS')
                        (obj, rel_pair, rel) = sgs[0]
                        obj = torch.argmax(obj, dim=1)
                        for i in range(obj.size(0)):
                            print(f'OBJECT {i}: ', obj[i])
                        print('DONE PRINTING OBJECTS')
                        is_printed=True

                '''
                image_list = None
                sgs = collate_sgs(sgs, cfg.MODEL.DEVICE)
                
                ''' 

                if is_main_process():
                    if not is_printed:
                        mistake_saver.add_mistake((image_ids[0], image_ids[1]), (sgs[0], sgs[1]), 231231, 'SG') 
                        mistake_saver.toHtml('/www')
                        is_printed = True
                
                ''' 

                listener_loss = 0
                gap_reward = 0
                avg_acc = 0
                num_correct = 0
                score_matrix = torch.zeros( (images.size(0), images.size(0)) )
                # fill score matrix
                for true_index, sg in enumerate(sgs):
                    acc = 0
                    detached_sg = (sg[0].detach().requires_grad_().float(), sg[1], sg[2].detach().requires_grad_().float() )
                    #scores = listener(sg, images)
                  #  with amp.disable_casts():
                    scores = listener(detached_sg, images)
                    score_matrix[true_index] = scores

                #print('Score matrix:', score_matrix)
                score_matrix = score_matrix.to(device)
                # fill loss matrix
                loss_matrix = torch.zeros( (2, images.size(0), images.size(0)), device=device)
                # sg centered scores
                for true_index in range(loss_matrix.size(1)):
                    row_score = score_matrix[true_index]
                    (true_scores, predicted_scores, binary) = format_scores(row_score, true_index, device)
                    loss_vec = listener_loss_func(true_scores, predicted_scores, binary)
                    loss_matrix[0][true_index] = loss_vec
                # image centered scores
                transposted_score_matrix = score_matrix.t()
                for true_index in range(loss_matrix.size(1)):
                    row_score = transposted_score_matrix[true_index]
                    (true_scores, predicted_scores, binary) = format_scores(row_score, true_index, device)
                    loss_vec = listener_loss_func(true_scores, predicted_scores, binary)
                    loss_matrix[1][true_index] = loss_vec


                print('iteration:', listener_iteration)
                sg_acc = 0
                img_acc = 0
                # calculate accuracy
                for i in range(loss_matrix.size(1)):
                    temp_sg_acc = 0
                    temp_img_acc = 0
                    for j in range(loss_matrix.size(2)):
                        if loss_matrix[0][i][i] > loss_matrix[0][i][j]:
                            temp_sg_acc += 1
                        else:
                            if is_main_process() and listener_iteration>=300 and listener_iteration % 25 ==0 and i != j:
                                detached_sg_i = (sgs[i][0].detach(), sgs[i][1], sgs[i][2].detach())
                                detached_sg_j = (sgs[j][0].detach(), sgs[j][1], sgs[j][2].detach())
                                mistake_saver.add_mistake((image_ids[i],image_ids[j]), (detached_sg_i,detached_sg_j), listener_iteration, 'SG')
                        if loss_matrix[1][i][i] > loss_matrix[1][j][i]:
                            temp_img_acc += 1  
                        else:
                            if is_main_process() and listener_iteration>=300 and listener_iteration % 25 == 0 and i!=j:
                                detached_sg_i = (sgs[i][0].detach(), sgs[i][1], sgs[i][2].detach())
                                detached_sg_j = (sgs[j][0].detach(), sgs[j][1], sgs[j][2].detach())
                                mistake_saver.add_mistake((image_ids[i],image_ids[j]), (detached_sg_i,detached_sg_j), listener_iteration, 'IMG')

                    temp_sg_acc = temp_sg_acc*100/(loss_matrix.size(1)-1)
                    temp_img_acc = temp_img_acc*100/(loss_matrix.size(1)-1)
                    sg_acc += temp_sg_acc
                    img_acc += temp_img_acc
                if is_main_process() and listener_iteration % 100  == 0 and listener_iteration >= 300:    
                    mistake_saver.toHtml('/www')
                    
                sg_acc /= loss_matrix.size(1)
                img_acc /= loss_matrix.size(1)

                avg_sg_acc = torch.tensor([sg_acc]).to(device)
                avg_img_acc = torch.tensor([img_acc]).to(device)
                # reduce acc over all gpus
                avg_acc = {'sg_acc' : avg_sg_acc, 'img_acc' : avg_img_acc}
                avg_acc_reduced = reduce_loss_dict(avg_acc)
                
                sg_acc = sum(acc for acc in avg_acc_reduced['sg_acc'])
                img_acc = sum(acc for acc in avg_acc_reduced['img_acc'])
                
                # log acc to wadb
                if is_main_process():
                    wandb.log({
                        "Train SG Accuracy": sg_acc.item(),
                        "Train IMG Accuracy": img_acc.item()
                    })
                
                
                sg_loss = 0
                img_loss = 0

                for i in range(loss_matrix.size(0)):
                    for j in range(loss_matrix.size(1)):
                        loss_matrix[i][j][j] = 0.
                        
                for i in range(loss_matrix.size(1)):
                    sg_loss += torch.max(loss_matrix[0][i])
                    img_loss += torch.max(loss_matrix[1][:][i])
                        
                sg_loss = sg_loss / loss_matrix.size(1)
                img_loss = img_loss / loss_matrix.size(1)
                sg_loss = sg_loss.to(device)
                img_loss = img_loss.to(device)

                loss_dict = {
                    'sg_loss' : sg_loss,
                    'img_loss' : img_loss
                }

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                sg_loss_reduced = loss_dict_reduced['sg_loss']
                img_loss_reduced = loss_dict_reduced['img_loss']
                if is_main_process():
                    wandb.log({"Train SG Loss": sg_loss_reduced})
                    wandb.log({"Train IMG Loss": img_loss_reduced})
                
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss=losses_reduced, **loss_dict_reduced)


            

                
                # Note: If mixed precision is not used, this ends up doing nothing
                # Otherwise apply loss scaling for mixed-precision recipe
                losses.backward()
                #with amp.scale_loss(losses, listener_optimizer) as scaled_losses:
                #    scaled_losses.backward()
                


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
                    """
                    print('Model before save')
                    print('****************************')
                    print(listener.gnn.conv1.node_model.node_mlp_1[0].weight)
                    print('****************************')
                    """
                    with amp.disable_casts():
                        listener_checkpointer.save("model_{:07d}".format(iteration))
                if iteration == max_iter:
                    listener_checkpointer.save("model_final")

                val_result = None # used for scheduler updating
                if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
                    logger.info("Start validating")
                    val_result = run_val(cfg, model, listener, val_data_loaders, distributed, logger)
                    (sg_loss, img_loss, sg_acc, img_acc) = val_result
                    
                    if is_main_process():
                        wandb.log({
                            "Validation SG Accuracy": sg_acc,
                            "Validation IMG Accuracy": img_acc,
                            "Validation SG Loss": sg_loss,
                            "Validation IMG Loss": img_loss,
                        })
                        
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
        if type(dataset_result) is not tuple:
            dataset_result = (dataset_result,)
        val_result.append(dataset_result)

    organized_result = [[val_result[i][j] for i in range(len(val_result))] for j in range(len(val_result[0]))]
    final_result = []
    for i in range(len(organized_result)):
        # support for multi gpu distributed testing
        gathered_result = all_gather(torch.tensor(organized_result[i]).cpu())
        gathered_result = [t.view(-1) for t in gathered_result]
        gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
        valid_result = gathered_result[gathered_result>=0]
        val_result = float(valid_result.mean())
        final_result.append(val_result)
        del gathered_result, valid_result
        torch.cuda.empty_cache()
    return tuple(final_result)

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
