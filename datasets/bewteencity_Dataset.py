# -*- coding: utf-8 -*-
import copy
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
NUM_CLASSES = 19

# colour map
label_colours = [
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [  0,   0,   0]] # the color of ignored label(-1) 
label_colours = list(map(tuple, label_colours))



class Group_Dataset(data.Dataset):
    def __init__(self,
                 args,
                 data_root_path=os.path.abspath('./datasets/Cityscapes'),
                 list_path=os.path.abspath('./datasets/between_city_list'),
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 class_16=False,
                 class_13=False,
                 group=0
                 ):
        """

        :param root_path:
        :param dataset:
        :param base_size:
        :param is_trainging:
        :param transforms:
        """
        self.args = args
        self.data_path=data_root_path
        self.list_path=list_path
        self.split=split
        self.base_size=base_size
        self.crop_size=crop_size

        self.base_size = self.base_size if isinstance(self.base_size, tuple) else (self.base_size, self.base_size)
        self.crop_size = self.crop_size if isinstance(self.crop_size, tuple) else (self.crop_size, self.crop_size)
        self.training = training

        self.random_mirror = args.random_mirror
        self.random_crop = args.random_crop
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur

        if group == 0:
            item_list_filepath = os.path.join(self.list_path, self.split+"0.txt")
        else:
            item_list_filepath = os.path.join(self.list_path, self.split+"1.txt")
        
        print(item_list_filepath)
        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainval")

        self.image_filepath = os.path.join(self.data_path, "leftImg8bit")

        self.gt_filepath = os.path.join(self.data_path, "gtFine")

        self.items = [id.strip() for id in open(item_list_filepath)]
        
        if self.args.city_name != 'None':
            self.items = [id for id in self.items if id.split('_')[1] == self.args.city_name ]

        ignore_label = -1
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        # In SYNTHIA-to-Cityscapes case, only consider 16 shared classes
        self.class_16 = class_16
        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_16id = {id:i for i,id in enumerate(synthia_set_16)}
        # In Cityscapes-to-NTHU case, only consider 13 shared classes
        self.class_13 = class_13
        synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_13id = {id:i for i,id in enumerate(synthia_set_13)}
        
        print("{} num images in Cityscapes {} set have been loaded.".format(len(self.items), self.split))
        if self.args.numpy_transform:
            print("use numpy_transform, instead of tensor transform!")

    def id2trainId(self, label, reverse=False, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32) # 
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        if self.class_16:
            label_copy_16 = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_16id.items():
                label_copy_16[label_copy == k] = v
            label_copy = label_copy_16
        if self.class_13:
            label_copy_13 = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_13id.items():
                label_copy_13[label_copy == k] = v
            label_copy = label_copy_13
        return label_copy

    def __getitem__(self, item):
        id = self.items[item]
        filename = id.split("train_")[-1].split("val_")[-1].split("test_")[-1]
        image_filepath = os.path.join(self.image_filepath, id.split("_")[0], id.split("_")[1])
        image_filename = filename + "_leftImg8bit.png"
        image_path = os.path.join(image_filepath, image_filename)
        image = Image.open(image_path).convert("RGB")

        gt_filepath = os.path.join(self.gt_filepath, id.split("_")[0], id.split("_")[1])
        gt_filename = filename + "_gtFine_labelIds.png"
        gt_image_path = os.path.join(gt_filepath, gt_filename)
        gt_image = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, item

    def _train_sync_transform(self, img, mask):
        '''
        :param image:  PIL input image
        :param gt_image: PIL input gt_image
        :return:
        '''
        if self.random_mirror: # 随机镜像
            # random mirror
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if mask: mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            crop_w, crop_h = self.crop_size

        if self.random_crop:  # 随机裁剪
            # random scale
            base_w , base_h = self.base_size
            w, h = img.size
            assert w >= h
            if (base_w / w) > (base_h / h):
                base_size = base_w 
                short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                base_size = base_h
                short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
                oh = short_size
                ow = int(1.0 * w * oh / h)

            img = img.resize((ow, oh), Image.BICUBIC)
            if mask: mask = mask.resize((ow, oh), Image.NEAREST)
            # pad crop
            if ow < crop_w or oh < crop_h:
                padh = crop_h - oh if oh < crop_h else 0
                padw = crop_w - ow if ow < crop_w else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                if mask: mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            # random crop crop_size
            w, h = img.size
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            if mask: mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        elif self.resize:
            img = img.resize(self.crop_size, Image.BICUBIC)
            if mask: mask = mask.resize(self.crop_size, Image.NEAREST)
        
        if self.gaussian_blur:
            # gaussian blur as in PSP
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(
                    radius=random.random()))
        # final transform
        if mask: 
            img, mask = self._img_transform(img), self._mask_transform(mask)
            return img, mask
        else:
            img = self._img_transform(img)
            return img

    def _val_sync_transform(self, img, mask):
        if self.random_crop:
            crop_w, crop_h = self.crop_size
            w, h = img.size
            if crop_w / w < crop_h / h:
                oh = crop_h
                ow = int(1.0 * w * oh / h)
            else:
                ow = crop_w
                oh = int(1.0 * h * ow / w)
            img = img.resize((ow, oh), Image.BICUBIC)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # center crop
            w, h = img.size
            x1 = int(round((w - crop_w) / 2.))
            y1 = int(round((h - crop_h) / 2.))
            img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        elif self.resize:
            img = img.resize(self.crop_size, Image.BICUBIC)
            mask = mask.resize(self.crop_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, image):
        if self.args.numpy_transform:
            image = np.asarray(image, np.float32)  
            image = image[:, :, ::-1]  # change to BGR
            image -= IMG_MEAN
            image = image.transpose((2, 0, 1)).copy() # (C x H x W)
            new_image = torch.from_numpy(image) # 
        else:
            image_transforms = ttransforms.Compose([
                ttransforms.ToTensor(),
                ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ])
            new_image = image_transforms(image)
        return new_image  # 

    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target

    def __len__(self):
        return len(self.items)


class Group0_DataLoader():
    def __init__(self, args, training=True):

        self.args = args
        data_set = Group_Dataset(args, 
                                data_root_path='./datasets/Cityscapes',
                                list_path='./datasets/between_city_list',
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training,
                                class_16=args.class_16,
                                class_13=args.class_13,
                                group=0)

        if (self.args.split == "train" or self.args.split == "trainval") and training:
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        else:
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)

        val_set = Group_Dataset(args, 
                            data_root_path='./datasets/Cityscapes',
                            list_path='./datasets/between_city_list',
                            split='val',
                            base_size=args.base_size,
                            crop_size=args.crop_size,
                            training=False,
                            class_16=args.class_16,
                            class_13=args.class_13,
                            group=0)
        self.val_loader = data.DataLoader(val_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size



class Group1_DataLoader():
    def __init__(self, args, training=True):

        self.args = args
        data_set = Group_Dataset(args, 
                                data_root_path='./datasets/Cityscapes',
                                list_path='./datasets/between_city_list',
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training,
                                class_16=args.class_16,
                                class_13=args.class_13,
                                group=1)

        if (self.args.split == "train" or self.args.split == "trainval") and training:
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        else:
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)

        val_set = Group_Dataset(args, 
                            data_root_path='./datasets/Cityscapes',
                            list_path='./datasets/between_city_list',
                            split='val',
                            base_size=args.base_size,
                            crop_size=args.crop_size,
                            training=False,
                            class_16=args.class_16,
                            class_13=args.class_13,
                            group=1)
        self.val_loader = data.DataLoader(val_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]

def inv_preprocess(imgs, num_images=1, img_mean=IMG_MEAN, numpy_transform=False):
    """Inverse preprocessing of the batch of images.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
      numpy_transform: whether change RGB to BGR during img_transform.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    if numpy_transform:
        imgs = flip(imgs, 1)
    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
    norm_ip(imgs, float(imgs.min()), float(imgs.max()))
    return imgs

def decode_labels(mask, num_images=1, num_classes=NUM_CLASSES):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)
    
name_classes = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'trafflight',
    'traffsign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'unlabeled'
]

def inspect_decode_labels(pred, num_images=1, num_classes=NUM_CLASSES, 
        inspect_split=[0.9, 0.8, 0.7, 0.5, 0.0], inspect_ratio=[1.0, 0.8, 0.6, 0.3]):
    """Decode batch of segmentation masks accroding to the prediction probability.
    
    Args:
      pred: result of inference.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
      inspect_split: probability between different split has different brightness.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.data.cpu().numpy()
    n, c, h, w = pred.shape
    pred = pred.transpose([0, 2, 3, 1])
    if n < num_images: 
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (w, h))
      pixels = img.load()
      for j_, j in enumerate(pred[i, :, :, :]):
          for k_, k in enumerate(j):
              assert k.shape[0] == num_classes
              k_value = np.max(softmax(k))
              k_class = np.argmax(k)
              for it, iv in enumerate(inspect_split):
                  if k_value > iv: break
              if iv > 0:
                pixels[k_,j_] = tuple(map(lambda x: int(inspect_ratio[it]*x), label_colours[k_class]))
      outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)