# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import datasets.transforms as T
from panopticapi.utils import rgb2id
from util.box_ops import masks_to_boxes

def make_carla_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


class CarlaPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        with open(ann_file, 'r') as f:
            self.carla = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.carla['images'] = sorted(self.carla['images'], key=lambda x: x['id'])
        # sanity check
        #if "annotations" in self.carla:
        #    for img, ann in zip(self.carla['images'], self.carla['annotations']):
        #        assert img['file_name'][:-4] == ann['file_name'][:-4]
        #print(self.carla["images"])
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        ann_info = self.carla['annotations'][idx]
        ann_path = Path(self.ann_folder) / ann_info['file_name']
        img_path = Path(self.img_folder) / ann_info['file_name'].replace('.png', '.jpg')

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if "segments_info" in ann_info:
            # CV2 to read it better
            #masks = np.asarray(Image.open(ann_path).convert('RGB'), dtype=np.uint32)
            #print(np.unique(masks[:,:,0]))
            masks = np.array(cv2.imread(str(ann_path)), dtype=np.uint32)
            #print(np.unique(masks[:,:,0]))
            #masks = cv2.imread(str(ann_path))
            #masks = rgb2id(masks)
            masks = masks[:, :, 0] + masks[:, :, 1] * 256 + masks[:, :, 2] * 256 ** 2

            ids = np.array([ann['id'] for ann in ann_info['segments_info']])
            masks = masks == ids[:, None, None]

            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]])
        if self.return_masks:
            target['masks'] = masks
        target['labels'] = labels

        target["boxes"] = masks_to_boxes(masks)
        

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        if "segments_info" in ann_info:
            for name in ['iscrowd', 'area']:
                target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.carla['images'])

    def get_height_and_width(self, idx):
        img_info = self.carla['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width


def build(image_set, args):
    img_folder_root = Path(args.carla_path)
    ann_folder_root = Path(args.carla_panoptic_path)
    assert img_folder_root.exists(), f'provided COCO path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided COCO path {ann_folder_root} does not exist'
    mode = 'panoptic'
    PATHS = {
        "train": ("train", Path("annotations") / f'{mode}_train.json'),
        "val": ("val", Path("annotations") / f'{mode}_val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder
    ann_folder = ann_folder_root / f'{mode}_{img_folder}'
    ann_file = ann_folder_root / ann_file

    dataset = CarlaPanoptic(img_folder_path, ann_folder, ann_file,
                           transforms=make_carla_transforms(image_set), return_masks=args.masks)

    return dataset
