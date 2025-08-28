from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization

from .contrastive_utils import MatchingHead, ContrastiveHead,SimpleMatchingHeadWithFocalLoss,MatchingHeadWithAllInBatchNegatives

import torch.nn as nn


class DropChannel(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        # x: [B, C, H, W]
        return self.drop(x)

import torch
import torch.nn as nn
from torch.nn import functional as F


class DropBlock(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.1):
        """
        块状丢弃模块 - 将块置零但不改变特征图大小
        Args:
            block_size: 要丢弃的块大小
            drop_prob: 丢弃概率
        """
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
    
    def forward(self, x):
        # 训练时才应用dropout
        if not self.training or self.drop_prob == 0:
            return x
        
        # x: [B, C, H, W]
        batch_size, channels, height, width = x.size()
        
        # 计算有效的块丢弃概率
        gamma = self.drop_prob * (height * width) / (self.block_size ** 2) / \
                ((height - self.block_size + 1) * (width - self.block_size + 1))
        
        # 生成伯努利mask
        mask_shape = (batch_size, channels, 
                     height - self.block_size + 1, 
                     width - self.block_size + 1)
        mask = torch.bernoulli(torch.ones(mask_shape, dtype=x.dtype, device=x.device) * gamma)
        
        # 使用max_pool2d扩展mask块
        # padding必须小于等于kernel_size的一半
        mask = F.max_pool2d(input=mask, 
                           kernel_size=self.block_size,
                           stride=1, 
                           padding=self.block_size // 2)  # 这里使用整除确保padding合法
        
        # 确保mask和x的尺寸完全匹配
        if mask.shape[2:] != x.shape[2:]:
            # 如果尺寸不匹配，进行调整
            pad_h = (height - mask.shape[2]) // 2
            pad_w = (width - mask.shape[3]) // 2
            if pad_h > 0 or pad_w > 0:
                mask = F.pad(mask, (pad_w, width - mask.shape[3] - pad_w,
                                   pad_h, height - mask.shape[2] - pad_h))
            else:
                # 如果mask比x大，裁剪
                mask = mask[:, :, :height, :width]
        
        # 反转mask（1表示保留，0表示丢弃）
        mask = 1 - mask
        
        # 应用mask
        out = x * mask
        
        # 归一化
        mask_mean = mask.mean()
        if mask_mean > 0:
            out = out / mask_mean
        
        return out




@MODELS.register_module()
class BEVFusionNoSwin(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()


####################################################ITM####################################################

        # self.itm_pre_head = SimpleMatchingHeadWithFocalLoss(
        #     in_channels_img=80,      # LiDAR 全局池化后的维度
        #     in_channels_lidar=256,
        #     hidden=128,
        #     use_projector=True,
        #     proj_dim=128
        # )

        self.itm_pre_head = MatchingHeadWithAllInBatchNegatives(
            in_channels_img=80, 
            in_channels_lidar=256,
            proj_dim=256  # 投影维度可以根据需求调整
        )



        # self.itc_post_head = ContrastiveHead(   ## ITC
        #     in_channels_img=80,          # image-BEV GAP 后维度
        #     in_channels_lidar=256,       # lidar-BEV GAP 后维度
        #     proj_dim=256,
        #     queue_size=16384,
        #     temperature=0.07,
        #     neg_k=4096,
        #     use_all_gather=True          # 多卡训练建议开
        # )

        self.itm_pre_weight = 0.4
        # self.itc_post_weight = 0.5

        self.drop_p =0.15
        self.drop_channel = DropChannel(p=self.drop_p)

        self.drop_block_p = 0.10  # 块状丢弃的概率，可以从配置文件传入
        self.block_size = 7      # 块的大小，可以从配置文件传入
        self.drop_block = DropBlock(drop_prob=self.drop_block_p, block_size=self.block_size)

        
####################################################ITM####################################################


    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()    # 4 6 3 256 704
        x = x.view(B * N, C, H, W).contiguous()

        # x = self.img_backbone(x)
        # x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)                        # torch.Size([bs, 80, 180, 180])

        pts_feature = self.extract_pts_feat(batch_inputs_dict)  # torch.Size([bs, 256, 180, 180])
        features.append(pts_feature)                            
        # features[0] img : torch.Size([bs, 80, 180, 180])
        # features[1] pts : torch.Size([bs, 256, 180, 180])
        # 此处为得到的所有特征，这里要做一次ITM，

####################################################ITM####################################################

        if self.training:  # 两个模态之间的ITM
            img_feat   = features[0]
            lidar_feat = features[1]

            img_feat = self.drop_block(img_feat)
            lidar_feat = self.drop_block(lidar_feat)

            img_feat   = self.drop_channel(img_feat)  # [B, 80, H, W]
            lidar_feat = self.drop_channel(lidar_feat)  # [B, 256, H, W]

            # 1) GAP 得到 [B, C]
            img_vec   = torch.nn.functional.adaptive_avg_pool2d(img_feat, 1).flatten(1)  # torch.Size([2, 80])
            lidar_vec = torch.nn.functional.adaptive_avg_pool2d(lidar_feat, 1).flatten(1)  # torch.Size([2, 256])
            # 2) 计算 ITM 损失（内部含投影与二分类头）
            self._loss_itm_pre = self.itm_pre_head(lidar_vec, img_vec) * self.itm_pre_weight

            # self._loss_itc_post = self.itc_post_head(lidar_vec, img_vec) * self.itc_post_weight    # 无法使用
            
        else:
            self._loss_itm_pre = None

####################################################ITM####################################################




        if self.fusion_layer is not None:
            x = self.fusion_layer(features)  # 输出：torch.Size([bs, 256, 180, 180]) 这里只是一个简单的线形层融合
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x) # 输出 ：[0] torch.Size([bs, 128, 180, 180]) [1] torch.Size([bs, 256, 90, 90])   使用的SECOND网络，和neck一起承担BEV encoder功能
        x = self.pts_neck(x)     # 输出 ：torch.Size([bs, 512, 180, 180])   使用的SECONDFPN网络，和neck一起承担BEV encoder功能



        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

####################################################ITM####################################################

        if self._loss_itm_pre is not None:
            losses['loss_itm_pre'] = self._loss_itm_pre

        # if self._loss_itc_post is not None:
        #     losses['loss_itc_post'] = self._loss_itc_post 


####################################################ITM####################################################


        return losses
