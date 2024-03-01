import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import torch.distributed as dist
sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.8")
from bilateralfilter import bilateralfilter, bilateralfilter_batch


class LIG_Loss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def cal_pos_logit(self,flags,prototype_flags,logits):
        
        mask_all_pos = flags == prototype_flags.T
        cls_index_ = torch.unique(flags)
        cls_index = cls_index_[cls_index_!=-1]
        cls_index = cls_index[cls_index!=0]
        prototype_cls_mask = torch.zeros_like(mask_all_pos).to(flags.device)
        for idx in cls_index:
            col = int(idx - 1)
            prototype_cls_mask[:,col] = 1

        logits_filtered_pos = logits * mask_all_pos
        # filter wrong labled pos pairs
        logits_filtered_all = logits * prototype_cls_mask
        # prevent nan 

        return logits_filtered_pos, logits_filtered_all, mask_all_pos


    def forward(self, output_q, prototypes,flags):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        output_q        :(20,DIM) 10*b,
        prototypes      :(20,DIM) 20
        flags           :(20,1)
        """
        num_cls = prototypes.shape[0]
        b = output_q.shape[0]
        prototypes_flag = torch.arange(1,num_cls+1).reshape(-1,1).to(flags.device) # 1,2,3,...,20 for VOC
        logits = torch.matmul(output_q, prototypes.T)
        logits = torch.div(logits , self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_all = logits - logits_max.detach()
        logits_all = logits

        logits_pos, logits_all, mask_pos_pos = self.cal_pos_logit(flags,prototypes_flag,logits_all)

        # compute log_prob
        exp_logits = torch.exp(logits_all)
        exp_logits = exp_logits * (exp_logits != 1.0000) # only calculate the appeared class
        log_prob = logits_pos - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        logits_pos_mask = logits_pos != 0
        log_prob = log_prob * logits_pos_mask

        loss = torch.tensor([0.0]).to(output_q.device)
        for idx in range(logits_pos.shape[0]):
            exp_logits_value = exp_logits.sum(1, keepdim=True)
            if exp_logits_value[idx] > 0:
                loss += log_prob[idx].sum()
                # loss_num += 1
            else:
                pass

        # compute mean of log-likelihood over positive
        if mask_pos_pos.sum() > 0:
            mean_log_prob_pos = loss / mask_pos_pos.sum()
        else:
            mean_log_prob_pos =  loss / logits_pos.shape[0]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        return loss


class LIL_Loss(nn.Module): 

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def cal_pos_logit(self,flags,queue_flags_all,logits,n_iter):

        mask_all_pos = flags == queue_flags_all.T
        
        idx_m1,_ = np.where(flags.cpu().numpy()==-1)
        # mask_all_pos[idx_0] = 0 # get rid of bkg
        mask_all_pos[idx_m1] = 0 # get rid of cooccurence
    
        # get rid of -1 cls 
        logits[idx_m1] = 0
        # logits[idx_0] = 0

        logits_filtered_pos = logits * mask_all_pos

        # filter wrong labled pos pairs
        if n_iter >= 0: # for stable demtermine 
            for i in range(len(flags)):
                mean_sim = logits_filtered_pos[i].mean()
                _pos_logits_index = torch.where(logits_filtered_pos[i] >= mean_sim)[0]
                _wrong_pos_logits_index = torch.where(logits_filtered_pos[i] < mean_sim)[0]
                if len(_wrong_pos_logits_index) / (len(_pos_logits_index) + 1e-6) > 10:
                    flags[i] = 0 # get rid of noise label 
                    logits_filtered_pos[i] = 0.0

        queue_index  = torch.where(queue_flags_all == -1)[0]
        logits[:,queue_index] = 0.0
        return logits_filtered_pos, logits, flags

    def forward(self, output_q, queue_all, flags, queue_flags_all,n_iter):
        b = output_q.shape[0]
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(output_q, queue_all[b:].T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_all = anchor_dot_contrast - logits_max.detach()
        # mask-out bkg and cooccurence
        logits_pos,logits_all,flags_revised = self.cal_pos_logit(flags,queue_flags_all[b:],logits_all,n_iter)
        # compute log_prob
        exp_logits = torch.exp(logits_all)
        exp_logits = exp_logits * (exp_logits != 1.0000) # only calculate the appeared class
        # only class pair is calculated
        log_prob = logits_pos - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        loss = torch.tensor([0.0]).to(output_q.device)
        for idx in range(logits_pos.shape[0]):
            num_nozero = torch.nonzero(logits_pos).size(0)
            exp_logits_value = exp_logits.sum(1, keepdim=True)
            if num_nozero > 0 and  exp_logits_value[idx] > 0 and log_prob[idx].sum() < 0:
                loss += log_prob[idx].sum() / num_nozero
            else:
                pass

        loss /= logits_pos.shape[0]

        # loss
        loss = - (self.temperature / self.base_temperature) * loss

        return loss,flags_revised


def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5

def get_masked_ptc_loss(inputs, mask):
    b, c, h, w = inputs.shape
    
    inputs = inputs.reshape(b, c, h*w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1,2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5*(1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum()+1)) + 0.5 * torch.sum(neg_mask * inputs_cos) / (neg_mask.sum()+1)
    return loss

def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5

def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    pred_prob = F.softmax(logit, dim=1)

    if img_box is not None:
        crop_mask = torch.zeros_like(pred_prob[:, 0, ...])
        for idx, coord in enumerate(img_box):
            crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1
    else:
        crop_mask = torch.ones_like(pred_prob[:, 0, ...])

    _img = torch.zeros_like(img)
    _img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
    _img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
    _img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor, recompute_scale_factor=True) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False, recompute_scale_factor=True)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor, recompute_scale_factor=True).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest', recompute_scale_factor=True)
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )

class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None
    
