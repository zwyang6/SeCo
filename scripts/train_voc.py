import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append("")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from model.losses import (get_masked_ptc_loss, get_seg_loss, get_energy_loss, 
                        LIG_Loss, LIL_Loss, DenseEnergyLoss)
from torch.nn.parallel import DistributedDataParallel
from model.PAR import PAR
from utils import imutils,evaluate
from utils.camutils import (cam_to_label, multi_scale_cam2, label_to_aff_mask, 
                            refine_cams_with_bkg_v2, assign_csc_tags, cam_to_roi_mask)
from utils.pyutils import AverageMeter, cal_eta, setup_logger
from utils.tbutils import make_grid_image, make_grid_label
from engine import build_dataloader, build_network, build_optimizer, build_validation
parser = argparse.ArgumentParser()
from torch.utils.tensorboard import SummaryWriter
torch.hub.set_dir("./pretrained")

### loss weight
parser.add_argument("--w_ptc", default=0.3, type=float, help="w_ptc")
parser.add_argument("--w_lil", default=0.5, type=float, help="w_lil")
parser.add_argument("--w_lig", default=0.5, type=float, help="w_lig")
parser.add_argument("--w_seg", default=0.12, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")

### training utils
parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=100, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")
parser.add_argument("--update_prototype", default=600, type=int, help="begin to update prototypes")
parser.add_argument("--cam2mask", default=10000, type=int, help="use mask from last layer")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")

### cam utils
parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--tag_threshold", default=0.2, type=int, help="filter cls tags")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75, 1.5), help="multi_scales for cam")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")

### knowledge extraction
parser.add_argument('--proto_m', default=0.9, type=float, help='momentum for computing the momving average of prototypes')
parser.add_argument("--temp_lil", default=0.08, type=float, help="temp")
parser.add_argument("--base_temp_lil", default=0.08, type=float, help="temp")
parser.add_argument("--temp_lig", default=0.1, type=float, help="temp")
parser.add_argument("--base_temp_lig", default=0.1, type=float, help="temp")
parser.add_argument("--momentum", default=0.999, type=float, help="momentum")
parser.add_argument('--ctc-dim', default=768, type=int, help='embedding dimension')
parser.add_argument('--moco_queue', default=4608, type=int, help='queue size; number of negative samples')
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

### log utils
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--tensorboard", default=True, type=bool, help="log tb")
parser.add_argument("--save_ckpt", default=False, action="store_true", help="save_ckpt")
parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--work_dir", default="w_outputs", type=str, help="w_outputs")
parser.add_argument("--log_tag", default="train_voc", type=str, help="train_voc")

### dataset utils
parser.add_argument("--data_folder", default='/data/ziqing/Jaye_Files/Dataset/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=64, type=int, help="crop_size for local view")
parser.add_argument('--ncrops', default=12, type=int, help='number of crops')
parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

### optimizer utils
parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    if args.local_rank == 0 and args.tensorboard == True:
        tb_logger = SummaryWriter(log_dir=args.tb_dir)

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    device = torch.device(args.local_rank)

    train_loader, train_sampler, val_loader = build_dataloader(args)
    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    model, param_groups = build_network(args)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    optim = build_optimizer(args,param_groups)
    logging.info('\nOptimizer: \n%s' % optim)

    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    get_lig_loss = LIG_Loss(temperature=args.temp_lig, base_temperature=args.base_temp_lig).cuda() 
    get_lil_loss = LIL_Loss(temperature=args.temp_lil, base_temperature=args.base_temp_lil).cuda() 
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    for n_iter in range(args.max_iters):
        global_step = n_iter + 1
        try:
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image = next(train_loader_iter)

        crops = [raw_image, w_image, s_image]
        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
        valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )

        ### descompose image to remove bias
        roi_mask_source = cams if n_iter >= args.cam2mask else cams_aux
        roi_mask = cam_to_roi_mask(roi_mask_source.detach(), cls_label=cls_label, img_box=img_box,ignore_mid=True,bkg_thre=args.bkg_thre,low_thre=args.low_thre, high_thre=args.high_thre,ignore_index=args.ignore_index)
        local_crops_k, _ = assign_csc_tags(images=crops[2], roi_mask=roi_mask, crop_num=args.ncrops-2, crop_size=args.local_crop_size, threshold=args.tag_threshold)
        local_crops_q, cls_flags_local= assign_csc_tags(images=crops[0], roi_mask=roi_mask, crop_num=args.ncrops-2, crop_size=args.local_crop_size,threshold=args.tag_threshold)
        roi_crops = crops[:2] + local_crops_q + local_crops_k
        cls_flags_local = cls_flags_local.reshape(-1,1).cuda()

        cls, segs, fmap, cls_aux, out_q, queue_feats_all, queue_flags_all, prototype = model(inputs,label_idx=label_idx,crops=roi_crops,cls_flags_local=cls_flags_local, n_iter=n_iter)

        ### cls loss & aux cls loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        ### tag-guided contrastive losses
        lil_loss, cls_flags_local_revised = get_lil_loss(out_q, queue_feats_all, cls_flags_local,queue_flags_all,n_iter)
        lig_loss = get_lig_loss(out_q, prototype, cls_flags_local_revised)

        ### seg_loss & reg_loss
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)

        ### aff loss from ToCo, https://github.com/rulixiang/ToCo
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # warmup
        if n_iter <= 1000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + 0.0 * lig_loss + 0.0 * lil_loss + 0.0 * seg_loss + 0.0 * reg_loss
        elif n_iter <= 2000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_lig * lig_loss + args.w_lil * lil_loss + 0.0 * seg_loss + 0.0 * reg_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_lig * lig_loss + args.w_lil * lil_loss + args.w_seg * seg_loss + args.w_reg * reg_loss

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),
            'lig_loss': lig_loss.item(),
            'lil_loss': lil_loss.item()
        })

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, lig_loss: %.4f, lil_loss: %.4f, seg_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, \
                                                        avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'), avg_meter.pop('lig_loss'),avg_meter.pop('lil_loss'), avg_meter.pop('seg_loss')))
                if tb_logger is not None:

                    grid_img1, grid_cam1 = make_grid_image(inputs.detach(), cams.detach(), cls_label.detach())
                    _, grid_cam_aux = make_grid_image(inputs.detach(), cams_aux.detach(), cls_label.detach())
                    grid_seg_gt1 = make_grid_label(refined_pseudo_label.detach())
                    grid_seg_pred = make_grid_label(torch.argmax(segs.detach(), dim=1))
                    tb_logger.add_image("visual/img1", grid_img1, global_step=global_step)
                    tb_logger.add_image("visual/cam1", grid_cam1, global_step=global_step)
                    tb_logger.add_image("visual/aux_cam", grid_cam_aux, global_step=global_step)
                    tb_logger.add_image("visual/seg_gt1", grid_seg_gt1, global_step=global_step)
                    tb_logger.add_image("visual/seg_pred", grid_seg_pred, global_step=global_step)

        if (n_iter + 1) % args.eval_iters == 0 and (n_iter + 1) >= 1:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
            val_cls_score, tab_results = build_validation(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    timestamp_1 = "{0:%Y-%m}".format(datetime.datetime.now())
    timestamp_2 = "{0:%d-%H-%M-%S}".format(datetime.datetime.now())
    exp_tag = f'{args.log_tag}_{timestamp_2}'
    args.work_dir = os.path.join(args.work_dir, timestamp_1, exp_tag)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")
    args.tb_dir = os.path.join(args.work_dir, "tensorboards")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)
        os.makedirs(args.tb_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)