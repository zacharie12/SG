# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.structures.image_list import to_image_list
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

# LISTENER IMPORTS
from maskrcnn_benchmark.listener.utils import format_scores, collate_sgs
from copy import deepcopy
import torchvision
from torchvision import transforms
import torch.nn.functional as F


def compute_on_dataset(model, data_loader, device, synchronize_gather=True, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                # relation detection needs the targets
                output = model(images.to(device), targets)
                print('SG output: ', output[0].extra_fields)

            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
    torch.cuda.empty_cache()
    return results_dict

def compute_listener_on_dataset(model, listener, data_loader, device, synchronize_gather=True, timer=None):
    model.eval()
    results_dict = {}
    listener_loss_func = torch.nn.MarginRankingLoss(margin=1, reduction='none')
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            images, targets, image_ids = batch

            images_list = deepcopy(images)
            #images_list = to_image_list(images_list, cfg.DATALOADER.SIZE_DIVISIBILITY).to(device)

            for i in range(len(images)):
                images[i] = images[i].unsqueeze(0)
                images[i] = F.interpolate(images[i], size=(224, 224), mode='bilinear', align_corners=False)
                images[i] = images[i].squeeze()

            images = torch.stack(images).to(device)

            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output=[]
                # relation detection needs the targets
                sgs=[]
                for i in range(len(images_list)):
                    model_input = to_image_list(images_list[i], cfg.DATALOADER.SIZE_DIVISIBILITY).to(device)
                    sg = model.forward(model_input, [targets[i]], ret_sg=True)
                    sgs.append(sg)
                
                sgs = collate_sgs(sgs, cfg.MODEL.DEVICE)

                accuracy = []
                listener_loss = 0

                score_matrix = torch.zeros( (images.size(0), images.size(0)) )
                # fill score matrix
                for true_index, sg in enumerate(sgs):
                    acc = 0
                    detached_sg = (sg[0].detach().requires_grad_(), sg[1], sg[2].detach().requires_grad_() )
                    #scores = listener(sg, images)
                    scores = listener(detached_sg, images)
                    score_matrix[true_index] = scores

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


                sg_acc = []
                img_acc = []
                # calculate accuracy
                for i in range(loss_matrix.size(1)):
                    temp_sg_acc = 0
                    temp_img_acc = 0
                    for j in range(loss_matrix.size(2)):
                        if loss_matrix[0][i][i] > loss_matrix[0][i][j]:
                            temp_sg_acc += 1
                        if loss_matrix[1][i][i] > loss_matrix[1][j][i]:
                            temp_img_acc += 1

                    temp_sg_acc = temp_sg_acc*100/(loss_matrix.size(1)-1)
                    temp_img_acc = temp_img_acc*100/(loss_matrix.size(1)-1)

                    sg_acc.append(temp_sg_acc)
                    img_acc.append(temp_img_acc)

                for i in range(loss_matrix.size(0)):
                    for j in range(loss_matrix.size(1)):
                        loss_matrix[i][j][j] = 0.

                sg_loss = []
                img_loss = []
                for i in range(loss_matrix.size(1)):
                    sg_loss.append(torch.max(loss_matrix[0][i]))
                    img_loss.append(torch.max(loss_matrix[1][:][i]))

                


            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            sg_loss = [o.to(cpu_device) for o in sg_loss]
            img_loss = [o.to(cpu_device) for o in img_loss]
            
            sg_acc = [torch.Tensor([acc]).to(cpu_device) for acc in sg_acc]
            img_acc = [torch.Tensor([acc]).to(cpu_device) for acc in img_acc]
            
        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: (sg_loss_i, img_loss_i, sg_acc_i, img_acc_i) \
                                    for img_id, sg_loss_i, img_loss_i, sg_acc_i, img_acc_i \
                                    in  zip(image_ids, sg_loss, img_loss, sg_acc, img_Acc)})

            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            results_dict.update(
                {img_id: (sg_loss_i, img_loss_i, sg_acc_i, img_acc_i) \
                for img_id, sg_loss_i, img_loss_i, sg_acc_i, img_acc_i \
                in  zip(image_ids, sg_loss, img_loss, sg_acc, img_Acc)}
            )
    torch.cuda.empty_cache()
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, synchronize_gather=True):
    if not synchronize_gather:
        all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return

    if synchronize_gather:
        predictions = predictions_per_gpu
    else:
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
    
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!"
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        logger=None,
):
    load_prediction_from_cache = cfg.TEST.ALLOW_LOAD_FROM_CACHE and output_folder is not None and os.path.exists(os.path.join(output_folder, "eval_results.pytorch"))
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    if load_prediction_from_cache:
        predictions = torch.load(os.path.join(output_folder, "eval_results.pytorch"), map_location=torch.device("cpu"))['predictions']
    else:
        predictions = compute_on_dataset(model, data_loader, device, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER, timer=inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if not load_prediction_from_cache:
        predictions = _accumulate_predictions_from_multiple_gpus(predictions, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)

    if not is_main_process():
        return -1.0

    #if output_folder is not None and not load_prediction_from_cache:
    #    torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(cfg=cfg,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    logger=logger,
                    **extra_args)

def listener_inference(
        cfg,
        model,
        listener,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        logger=None,
):
    load_prediction_from_cache = cfg.TEST.ALLOW_LOAD_FROM_CACHE and output_folder is not None and os.path.exists(os.path.join(output_folder, "eval_results.pytorch"))
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    if load_prediction_from_cache:
        predictions = torch.load(os.path.join(output_folder, "eval_results.pytorch"), map_location=torch.device("cpu"))['predictions']
    else:
        predictions = compute_listener_on_dataset(model, listener, data_loader, device, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER, timer=inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if not load_prediction_from_cache:
        predictions = _accumulate_predictions_from_multiple_gpus(predictions, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)

    if not is_main_process():
        return -1.0

    #if output_folder is not None and not load_prediction_from_cache:
    #    torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    predictions = [[predictions[i][j] for i in range(len(predictions))] for j in range(len(predictions[0]))]
    print('Predictions: ', predictions, len(predictions))
    
    output_list = []
    for i, result in enumerate(predictions):
        out_sum = sum(result)
        out_sum = out_sum / (len(predictions[i]) 
        output_list.append(out_sum)
    
    return output_list

