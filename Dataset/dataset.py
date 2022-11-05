from torch.utils.data import Dataset,DataLoader
import skimage
import numpy as np
import imageio
import os
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

image_h = 480
image_w = 640

class RGBD_NYU_dataset(Dataset):
    def __init__(self,rgb_path:str,dep_path:str,lab_path:str,mode:str,transform=None):
        self.rgb_path = rgb_path
        self.dep_path = dep_path
        self.lab_path = lab_path
        self.rgb_dir = os.listdir(self.rgb_path)
        self.dep_dir = os.listdir(self.dep_path)
        self.lab_dir = os.listdir(self.lab_path)

        self.mode = mode
        self.transform = transform
    def __len__(self):
        return len(self.rgb_dir)

    def __getitem__(self, item):
        rgb_name = os.path.join(self.rgb_path, self.rgb_dir[item])
        dep_name = os.path.join(self.dep_path, self.dep_dir[item])
        lab_name = os.path.join(self.lab_path, self.lab_dir[item])

        #opencv read type:H,W,C
        lab = imageio.imread(lab_name)
        dep = imageio.imread(dep_name)
        rgb = imageio.imread(rgb_name)

        sample = {'image': rgb, 'depth': dep, 'label': lab}

        if self.mode == "train":
            sample = self.transform(sample)

        return sample


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm_depth(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class scaleNorm_hha(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        depth = skimage.transform.resize(depth, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)

        # Nearest-neighbor
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale_depth(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale_hha(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)

        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)

        # Nearest-neighbor
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop_depth(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomCrop_hha(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w, :],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.75:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        elif random.random() <= 0.75 and random.random() > 0.5:
            image = np.flipud(image).copy()
            depth = np.flipud(depth).copy()
            label = np.flipud(label).copy()

        elif random.random() <= 0.5 and random.random() > 0.25  :
            image = np.flipud(image).copy()
            depth = np.flipud(depth).copy()
            label = np.flipud(label).copy()
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize_depth(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class Normalize_hha(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        depth = depth / 255
        depth = torchvision.transforms.Normalize(mean=[0.538, 0.444, 0.439],
                                                 std=[0.211, 0.254, 0.146])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor_depth(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Generate different label scales
        label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label/1.0).float(),
                'label2': torch.from_numpy(label2/1.0).float(),
                'label3': torch.from_numpy(label3/1.0).float(),
                'label4': torch.from_numpy(label4/1.0).float(),
                'label5': torch.from_numpy(label5/1.0).float()}


class ToTensor_hha(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Generate different label scales
        label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = depth.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label/1.0).float(),
                'label2': torch.from_numpy(label2/1.0).float(),
                'label3': torch.from_numpy(label3/1.0).float(),
                'label4': torch.from_numpy(label4/1.0).float(),
                'label5': torch.from_numpy(label5/1.0).float()}


def transforms_hha():
    return transforms.Compose([scaleNorm_hha(),
                               RandomScale_hha((1.0, 1.6)),
                               RandomHSV((0.9, 1.1),
                                         (0.9, 1.1),
                                         (25, 25)),
                               RandomCrop_hha(image_h, image_w),
                               RandomFlip(),
                               ToTensor_hha()])
def transforms_depth():
    return transforms.Compose([scaleNorm_depth(),
                               RandomScale_depth((1.0, 2.0)),
                               RandomHSV((0.9, 1.1),
                                         (0.9, 1.1),
                                         (25, 25)),
                               RandomCrop_depth(image_h, image_w),
                               RandomFlip(),
                               ToTensor_depth()])


if __name__ =='__main__':
    rgbpath = r'datasets\NYUV2\train_rgb'
    deppath = r'datasets\NYUV2\train_hha'
    labpath = r'datasets\NYUV2\train_class40'

    dataset = RGBD_NYU_dataset(rgb_path=rgbpath, dep_path=deppath, lab_path=labpath, mode= 'train',
                               transform=transforms_hha())



    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    device = 'cpu'
    for batch_idx, sample in enumerate(dataloader):
        image = sample['image'].to(device)
        depth = sample['depth'].to(device)
        # target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
        target_scales = [sample[s].to(device) for s in ['label']]
        print(image.shape,depth.shape, target_scales[0].shape)

        rgb = image.squeeze()
        rgb = rgb.permute(1,2,0)
        rgb = np.array(rgb).astype(np.uint8)
        plt.figure("dog1")

        dep = depth.squeeze()
        dep = dep.permute(1, 2, 0)
        dep = np.array(dep).astype(np.uint16)


        lab = target_scales[0].squeeze()
        lab = np.array(lab).astype(np.uint16)


        plt.imshow(rgb/255)
        plt.show()
        plt.imshow(dep/255)
        plt.show()
        plt.imshow(lab*5/255)
        plt.show()

        dep = depth.squeeze()
        dep = np.array(dep).astype(np.uint16)
        lab0 = target_scales[0].squeeze()
        lab0 = np.array(lab0*15).astype(np.uint8)
        lab1 = target_scales[1].squeeze()
        lab1 = np.array(lab1*15).astype(np.uint8)
        lab2 = target_scales[2].squeeze()
        lab2 = np.array(lab2*15).astype(np.uint8)
        lab3 = target_scales[3].squeeze()
        lab3 = np.array(lab3*15).astype(np.uint8)
        lab4 = target_scales[4].squeeze()
        lab4 = np.array(lab4*15).astype(np.uint8)

        cv2.imshow('rgb', rgb)
        cv2.imshow('dep', dep)
        cv2.imshow('lab0', lab0)
        cv2.imshow('lab1', lab1)
        cv2.imshow('lab2', lab2)
        cv2.imshow('lab3', lab3)
        cv2.imshow('lab4', lab4)
        cv2.waitKey(0)




