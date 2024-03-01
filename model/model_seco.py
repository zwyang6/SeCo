import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
from . import decoder
from . import proj_head

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class network(nn.Module):
    def __init__(self, args,backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.init_momentum = init_momentum
        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)
        self.encoder_k = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        self.proj_head = proj_head.DINOHead(in_dim=self.encoder.embed_dim, out_dim=768)
        self.proj_head_t = proj_head.DINOHead(in_dim=self.encoder.embed_dim, out_dim=768,)

        for param, param_t in zip(self.proj_head.parameters(), self.proj_head_t.parameters()):
            param_t.data.copy_(param.data)  # initialize teacher with student
            param_t.requires_grad = False  # do not update by gradient

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4 
        self.pooling = F.adaptive_max_pool2d

        self.decoder = decoder.ASPP(in_planes=self.in_channels[-1], out_planes=self.num_classes,)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        
        self.register_buffer("prototypes", torch.randn(args.num_classes - 1,args.ctc_dim))
        self.register_buffer("queue", torch.randn(args.moco_queue, args.ctc_dim))
        self.register_buffer("queue_label_flag", torch.zeros(args.moco_queue,1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0) 

    ### extract category knowledge 
    @torch.no_grad()
    def update_prototype(self,embeds,label_idx):
        """embeds      : feats from encoder (b,DIM)
           label_idx   : the class index for embeds (b,1)"""
        for feat, label in zip(concat_all_gather(embeds), concat_all_gather(label_idx)):
            if label != -1:
                self.prototypes[label-1] = self.prototypes[label-1]*self.args.proto_m + (1-self.args.proto_m)*feat   
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        prototypes = self.prototypes.clone().detach()
        return prototypes

    ### maintain local semntic reservoir with tag pool 
    def maintain_reservoir(self, crops, cls_flags_local,n_iter=None):
        global_view = crops[:2]
        local_view_q = crops[2:self.args.ncrops]
        local_view_k = crops[self.args.ncrops:]

        n_g = len(global_view)
        n_l = len(local_view_q)
        batch_size = global_view[0].shape[0]

        local_inputs_q = torch.cat(local_view_q, dim=0)
        local_inputs_k = torch.cat(local_view_k, dim=0)

        global_output_s = self.encoder.forward_features(torch.cat(global_view, dim=0))[0]
        local_output_q = self.encoder.forward_features(local_inputs_q)[0]
        global_output_s = global_output_s.reshape(n_g,batch_size,-1).permute(1,0,2) # 4,2,dim
        local_output_q = local_output_q.reshape(n_l,batch_size,-1).permute(1,0,2) # 4,10,dim
        output_s_q = torch.cat((global_output_s, local_output_q), dim=1).reshape((batch_size*(n_l+n_g)),-1)
        output_s_q = self.proj_head(output_s_q)

        self._momentum_update_key_encoder(self.args)
        local_output_k = self.encoder_k.forward_features(local_inputs_k)[0]
        local_output_k = local_output_k.reshape(n_l,batch_size,-1).permute(1,0,2) # 4,10,dim

        # global and local_k are sent to queue
        output_s_k = torch.cat((global_output_s, local_output_k), dim=1).reshape((batch_size*(n_l+n_g)),-1)
        # shuffle for making use of BN
        output_s_k, idx_unshuffle = self._batch_shuffle_ddp(output_s_k)
        self._EMA_update_encoder_teacher(n_iter)
        output_s_k = self.proj_head_t(output_s_k)
        # undo shuffle
        output_s_k = self._batch_unshuffle_ddp(output_s_k, idx_unshuffle)
        # dequeue and enqueue
        self._dequeue_and_enqueue(output_s_k, cls_flags_local, self.args)

        # queue cls flags
        queue_flags_all = torch.cat((cls_flags_local, cls_flags_local, self.queue_label_flag.clone().detach()), dim=0)
        queue_feats_all = torch.cat((output_s_q, output_s_k, self.queue.clone().detach()), dim=0)
    
        return output_s_k, output_s_q, queue_flags_all, queue_feats_all

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_label_flag[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
   
    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        momentum = self.init_momentum
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self, n_iter=None):
        ## no scheduler here
        momentum = self.init_momentum
        for param, param_t in zip(self.proj_head.parameters(), self.proj_head_t.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)

        for param in list(self.proj_head.parameters()):
            param_groups[2].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x, label_idx=None, crops=None,cls_flags_local=None, n_iter=None,cam_only=False):
        
        cls_token, _x, x_aux = self.encoder.forward_features(x)
        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size
        _x4 = self.to_2D(_x, h, w)
        _x_aux = self.to_2D(x_aux, h, w)

        if cam_only:
            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()
            return cam_aux, cam
                
        cls_aux = self.pooling(_x_aux, (1,1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1,1))
        cls_x4 = self.classifier(cls_x4)

        cls_x4 = cls_x4.view(-1, self.num_classes-1)
        cls_aux = cls_aux.view(-1, self.num_classes-1)

        seg = self.decoder(_x4)

        
        if crops is None:
            return cls_x4, seg, _x4, cls_aux
        else:
            if n_iter >= self.args.update_prototype: 
                prototypes = self.update_prototype(cls_token.contiguous(),label_idx.contiguous()).to(x.device)
            else:
                prototypes = self.prototypes
            _, output_s_q, queue_flags_all, queue_feats_all = self.maintain_reservoir(crops, cls_flags_local,n_iter)   
            return cls_x4, seg, _x4, cls_aux, output_s_q, queue_feats_all, queue_flags_all, prototypes
