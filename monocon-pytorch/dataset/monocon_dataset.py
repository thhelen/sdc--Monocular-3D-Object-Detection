
import os
import sys
import torch
import numpy as np

from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from transforms import *
from dataset.base_dataset import BaseKITTIMono3DDataset
from PIL import Image
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torchvision.transforms import GaussianBlur
from torchvision import transforms


DEFAULT_FILTER_CONFIG = {
    'min_height': 25,
    'min_depth': 2,
    'max_depth': 65,
    'max_truncation': 0.5,
    'max_occlusion': 2,
}
# class UpperFogEffect(transforms.RandomApply):
#     def __init__(self, p, blur_radius_range=(5, 15), fog_region=(0.0, 0.5)):
#         blur_transform = transforms.Compose([
#             transforms.Lambda(lambda x: self.apply_blur(x, blur_radius_range, fog_region))
#         ])
#         transforms_list = [transforms.Lambda(lambda x: x)]
#         super().__init__([transforms.Compose(transforms_list + [blur_transform])], p=p)

#     def apply_blur(self, img, radius_range, region):
#         img = data_dict['img']
#         h, w, _ = img.shape
#         start, end = int(h * region[0]), int(h * region[1])
        
#         # Apply blur only to the top half of the image
#         blurred_top = Image.fromarray(img[:start, ...])
#         blurred_top = blurred_top.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(*radius_range)))
#         img[:start, ...] = np.array(blurred_top)
        
#         return img

default_train_transforms = [
    PhotometricDistortion(
        brightness_delta=30,
        contrast_range=(0.5, 1.0),
        saturation_range=(0.5, 1.0),
        hue_delta=15),
    RandomShift(prob=0.5, shift_range=(-32, 32), hide_kpts_in_shift_area=True),
    RandomHorizontalFlip(prob=0.5),
    RandomCrop3D(prob=0.5, crop_size=(320, 960), hide_kpts_in_crop_area=True),
    # UpperFogEffect(p=0.5, blur_radius_range=(5, 15)),

    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    Pad(size_divisor=32),
    ToTensor(),
]


default_test_transforms = [
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    Pad(size_divisor=32),
    ToTensor(),
]


class MonoConDataset(BaseKITTIMono3DDataset):
    def __init__(self, 
                 base_root: str, 
                 split: str,
                 max_objs: int = 30,
                 transforms: List[BaseTransform] = None,
                 filter_configs: Dict[str, Any] = None,
                 **kwargs):
        
        super().__init__(base_root=base_root, split=split, **kwargs)
        
        self.max_objs = max_objs
        
        if transforms is None:
            if (split == 'train'):
                transforms = default_train_transforms
            else:
                transforms = default_test_transforms
        self.transforms = Compose(transforms)
        
        if filter_configs is None:
            filter_configs = DEFAULT_FILTER_CONFIG
        else:
            cfg_keys = list(filter_configs.keys())
            flag = all([(key in DEFAULT_FILTER_CONFIG) for key in cfg_keys])
            assert flag, f"Keys in argument 'configs' must be one in {list(DEFAULT_FILTER_CONFIG.keys())}."
            
        for k, v in filter_configs.items():
            setattr(self, k, v)
        self.filter_configs = filter_configs
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        
        image, img_metas = self.load_image(idx)
        # print("###")
        # print(image.shape)
        # image12 = Image.fromarray(image)
        # image12.save('/home/jupyter/monocon-pytorch/before.png')
        calib = self.load_calib(idx)
        if self.split == 'test':
            result_dict = {
                'img': image,
                'img_metas': img_metas,
                'calib': calib}
            
            result_dict = self.transforms(result_dict)
            return result_dict
        # print(image)
        # print(image_metas)
        # print(self.load_label(0))
        # Raw State: Cam0 + Bottom-Center + Global Yaw
        # Converted to Cam2 + Local Yaw
        raw_labels = self.load_label(idx)
     
        raw_labels.convert_cam(src_cam=0, dst_cam=2)
        raw_labels.convert_yaw(src_type='global', dst_type='local')
        
        new_labels = self._create_empty_labels()
        
        input_hw = image.shape[:2]
        for obj_idx, raw_label in enumerate(raw_labels):
            
            # Base Properties
            occ = raw_label.occlusion
            trunc = raw_label.truncation
            
            if (occ > self.max_occlusion) or (trunc > self.max_truncation):
                continue
            
            
            # 2D Box Properties
            gt_bbox = raw_label.box2d
            bbox_height = (gt_bbox[3] - gt_bbox[1])
            gt_label = raw_label.cls_num
            
            if bbox_height < self.min_height:
                continue
            
            
            # 3D Box Properties
            gt_bbox_3d = np.concatenate([
                raw_label.loc,
                raw_label.dim,
                np.array([raw_label.ry])
            ], axis=0)
            gt_label_3d = gt_label
            
            
            # 2D-3D Properties
            projected = raw_label.projected_center
            center2d, depth = projected[:-1], projected[-1]
            
            if not (self.min_depth <= depth <= self.max_depth):
                continue
            
            
            # 2D Keypoints
            keypoints = raw_label.projected_kpts            # (9, 3) / 8 Corners + 1 Center
            for k_idx, keypoint in enumerate(keypoints):
                kptx, kpty, _ = keypoint
                
                is_kpt_in_image = (0 <= kptx <= input_hw[1]) and (0 <= kpty <= input_hw[0])
                if is_kpt_in_image:
                    keypoints[k_idx, 2] = 2
            
            
            # Add Labels
            new_labels['gt_bboxes'][obj_idx, :] = gt_bbox
            new_labels['gt_labels'][obj_idx] = gt_label
            
            new_labels['gt_bboxes_3d'][obj_idx, :] = gt_bbox_3d
            new_labels['gt_labels_3d'][obj_idx] = gt_label_3d
            
            new_labels['centers2d'][obj_idx] = center2d
            new_labels['depths'][obj_idx] = depth
            
            new_labels['gt_kpts_2d'][obj_idx] = keypoints[:, :2].reshape(-1)
            new_labels['gt_kpts_valid_mask'][obj_idx] = keypoints[:, 2]
            
            new_labels['mask'][obj_idx] = True
        
        result_dict = {
            'img': image,
            'img_metas': img_metas,
            'calib': calib,
            'label': new_labels}
        
        # with open('og_image.npy', 'wb') as f:
        #     np.save(f, result_dict['img'])
        
        result_dict = self.transforms(result_dict)

#         with open('og_trans.npy', 'wb') as f:
#             np.save(f, result_dict['img'].numpy())

#         transform2 = A.Compose([
#             #A.ColorJitter(brightness=0.3, p=1), contrast=0.5, saturation=0.5, hue=0.1, p=1),
#             A.RandomBrightnessContrast(brightness_limit=(-0.25, -0.15), p=0.3),
#             A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            
#             # A.FancyPCA(alpha=0.1, p=0.5),
#             # A.ToGray(p=0.2),
#             A.RandomFog(fog_coef_lower=0.15, fog_coef_upper=0.25, alpha_coef=0.07, p=0.3),
#             #A.augmentations.geometric.resize.Resize(height=375, width=1240, p=1),
#             RandomShift(prob=0.5, shift_range=(-32, 32),hide_kpts_in_shift_area=True),
#             RandomHorizontalFlip(prob=0.5),
#             RandomCrop3D(prob=0.5, crop_size=(320, 960), hide_kpts_in_crop_area=True),
#             Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
#             Pad(size_divisor=32),
#             ToTensorV2(),
#         ])
        
        #transformed_image = transform2(image=result_dict['img'])
        # with open('new_trans.npy', 'wb') as f:
        #     np.save(f, transformed_image['image'].numpy())
            
         #result_dict['img'] = transform2(image=result_dict['img'])        #transformed_image['image'].type(torch.FloatTensor)
        # print("################")
        # print(type(transformed_image))
        # print(transformed_image)
        # if idx %2 ==0:
            
        # image = Image.fromarray(transformed_image['image'].numpy().transpose(1, 2, 0))
        # image.save('/home/jupyter/monocon-pytorch/test-image.png')
        # print('saved')
        # sys.exit()
        return result_dict
            
    def _create_empty_labels(self) -> Dict[str, np.ndarray]:
        annot_dict = {
            'gt_bboxes': np.zeros((self.max_objs, 4), dtype=np.float32),
            'gt_labels': np.zeros(self.max_objs, dtype=np.uint8),
            'gt_bboxes_3d': np.zeros((self.max_objs, 7), dtype=np.float32),
            'gt_labels_3d': np.zeros(self.max_objs, dtype=np.uint8),
            'centers2d': np.zeros((self.max_objs, 2), dtype=np.float32),
            'depths': np.zeros(self.max_objs, dtype=np.float32),
            'gt_kpts_2d': np.zeros((self.max_objs, 18), dtype=np.float32),
            'gt_kpts_valid_mask': np.zeros((self.max_objs, 9), dtype=np.uint8),
            'mask': np.zeros((self.max_objs,), dtype=np.bool_)}
        return annot_dict
    
    @staticmethod
    def collate_fn(batched: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Merge Image
        merged_image = torch.cat([d['img'].unsqueeze(0) for d in batched], dim=0)
        
        # Merge Image Metas
        img_metas_list = [d['img_metas'] for d in batched]
        merged_metas = {k: [] for k in img_metas_list[0].keys()}
        
        for img_metas in img_metas_list:
            for k, v in img_metas.items():
                merged_metas[k].append(v)
        
        # Merge Calib
        merged_calib = [d['calib'] for d in batched]
        if 'label' not in batched[0]:
            return {'img': merged_image, 
                    'img_metas': merged_metas, 
                    'calib': merged_calib}
        
        # Merge Label
        label_list = [d['label'] for d in batched]
        
        label_keys = label_list[0].keys()
        merged_label = {k: None for k in label_keys}
        for key in label_keys:
            merged_label[key] = torch.cat([torch.tensor(label[key]) for label in label_list], dim=0)

        return {'img': merged_image, 
                'img_metas': merged_metas, 
                'calib': merged_calib, 
                'label': merged_label}
