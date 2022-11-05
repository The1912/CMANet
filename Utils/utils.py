import time

import numpy as np
from torch import nn
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Dataset.dataset import RGBD_NYU_dataset, Normalize_depth, Normalize_hha, ToTensor_depth, ToTensor_hha
from Utils.transforms import transforms_hha
from Utils.metrics import SegmentationMetric
from Utils.load import load_checkpoint
import torchvision.transforms as transforms

med_frq_sunrgbd = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
                   0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
                   2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
                   0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
                   1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
                   4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
                   3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
                   0.750738, 4.040773]

med_frq_NYU2 = [0.71142, 0.77194, 0.81391, 0.94143, 0.95208, 1.07680, 1.14535, 1.14909, 1.10488,
                1.17025, 1.15129, 1.38882, 1.28413, 1.63873, 1.44600, 1.64278, 1.72670, 1.87923,
                1.57468, 2.17806, 1.81955, 1.33333, 2.46872, 2.50839, 2.41232, 3.29888, 3.56601,
                3.05714, 3.02347, 4.33856, 3.85831, 4.45739, 4.07171, 4.66439, 4.24873, 5.12041,
                5.27822, 1.09183, 1.18313, 0.87580]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

label_colours_NYU = [(0, 0, 0),
                     # 0=background
                     (148, 65, 137), (255, 116, 69), (86, 156, 137),
                     (202, 179, 158), (155, 99, 235), (161, 107, 108),
                     (133, 160, 103), (76, 152, 126), (84, 62, 35),
                     (44, 80, 130), (31, 184, 157), (101, 144, 77),
                     (23, 197, 62), (141, 168, 145), (142, 151, 136),
                     (115, 201, 77), (100, 216, 255), (57, 156, 36),
                     (88, 108, 129), (105, 129, 112), (42, 137, 126),
                     (155, 108, 249), (166, 148, 143), (81, 91, 87),
                     (100, 124, 51), (73, 131, 121), (157, 210, 220),
                     (134, 181, 60), (221, 223, 147), (123, 108, 131),
                     (161, 66, 179), (163, 221, 160), (31, 146, 98),
                     (99, 121, 30), (49, 89, 240), (116, 108, 9),
                     (161, 176, 169), (80, 29, 135), (177, 105, 197),
                     (139, 110, 246), (86, 101, 115), (153, 163, 164),
                     (220, 118, 51)]

test_rgbpath = r'/root/autodl-tmp/datasets/NYU2/test_rgb'
test_hhapath = r'/root/autodl-tmp/datasets/NYU2/test_hha'
test_deppath = r'/root/autodl-tmp/datasets/NYU2/test_depth'
test_labpath = r'/root/autodl-tmp/datasets/NYU2/test_class40'


def predit2label(predit):
    score = F.softmax(predit, dim=1)
    label = torch.argmax(score, dim=1)
    return label


def color_label(label):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours_NYU[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def print_log(epoch, lr, time, itr_idx, total_itr, loss, pa, mpa, miou):
    if itr_idx + 1 == total_itr:
        print('\r', 'Train Epoch: {:>4} Itera: {:>4}/{:>4} '
                    'Lr = {:.4e} Time = {:.2f}s '
                    'Loss = {:.4f} '
                    'Pa = {:.2%} mPa = {:.2%} mIoU = {:.2%}'.format(epoch, itr_idx + 1, total_itr, lr, time, loss,
                                                                    pa, mpa, miou)
              , end='\n')
    elif itr_idx + 1 < total_itr:
        print('\r', 'Train Epoch: {:>4} Itera: {:>4}/{:>4} '
                    'Lr = {:.4e} Time = {:.2f}s '
                    'Loss = {:.4f} '
                    'Pa = {:.2%} mPa = {:.2%} mIoU = {:.2%}'.format(epoch, itr_idx + 1, total_itr, lr, time, loss,
                                                                    pa, mpa, miou)
              , end=' ')

    else:
        print('print error!')
        os._exit(0)


def save_metrics(epoch, lr, time, loss, pa, mpa, miou):
    f = open('Train_metrics.txt', "a")
    f.write('Train Epoch: {:>4}  '
            'Lr = {:.4e} Time = {:.2f}s '
            'Loss = {:.4f} '
            'Pa = {:.2%} mPa = {:.2%} mIoU = {:.2%} \n'.format(epoch, lr, time, loss, pa, mpa, miou))


def test_metrics(epoch, net, depth_type, numclass):
    with torch.no_grad():
        print('test in epoch {}'.format(epoch))
        device = 'cuda'

        if depth_type == 'hha':
            dataset = RGBD_NYU_dataset(rgb_path=test_rgbpath,
                                       dep_path=test_hhapath,
                                       lab_path=test_labpath,
                                       mode='train',
                                       transform=transforms.Compose([ToTensor_hha(),
                                                                     Normalize_hha()]))

            dataloader = DataLoader(dataset,
                                    batch_size=32,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=False)

        elif depth_type == 'depth':
            dataset = RGBD_NYU_dataset(rgb_path=test_rgbpath,
                                       dep_path=test_deppath,
                                       lab_path=test_labpath,
                                       mode='train',
                                       transform=transforms.Compose([ToTensor_depth(),
                                                                     Normalize_depth()]))

            dataloader = DataLoader(dataset,
                                    batch_size=32,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=False)

        else:
            print('no {} depth type'.format(depth_type))
            os._exit(0)

        net.eval()

        strat_time = time.time()
        metricclass = SegmentationMetric(numClass=numclass)
        for batch_idx, sample in enumerate(dataloader):
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
            predit = net(image, depth)
            metricclass.addBatch(imgPredict=predit2label(predit[0].to('cpu')), imgLabel=target_scales[0].to('cpu') - 1)

        Pa = metricclass.pixelAccuracy()
        mPa = metricclass.meanPixelAccuracy()
        cPa = metricclass.classPixelAccuracy()
        mIOU = metricclass.meanIntersectionOverUnion()
        IoU = metricclass.IntersectionOverUnion()

        end_time = time.time()

        print('test time {}s'.format(end_time - strat_time))

        f = open('Test_metrics.txt', "a")
        f.write('Epochs {} \n'.format(epoch))

        f.write('pixel_accuracy :  \n')  # write pixel accuracy
        f.write('{}\n'.format(Pa))

        f.write('mean_pixel_accuracy :\n')  # write mean pixel accuracy
        f.write('{}\n'.format(mPa))

        f.write('class_pixel_accuracy :\n')  # write class pixel accuracy
        f.write('{}\n'.format(cPa))

        f.write('mean_IOU :\n')  # write mean IOU
        f.write('{}\n'.format(mIOU))

        f.write('class_IOU :\n')  # write class IOU
        f.write('{}\n\n\n'.format(IoU))

        print('Epoch {} metrics saved!'.format(epoch))


        net.train()


def depth_mode(mode, batchsize, rgb_path, dep_path, hha_path, lab_path):
    if mode == 'hha':
        dataset = RGBD_NYU_dataset(rgb_path=rgb_path,
                                   dep_path=hha_path,
                                   lab_path=lab_path,
                                   mode='train',
                                   transform=transforms_hha())

        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=False,
                                drop_last=True)


    # depth type is hha
    elif mode == 'depth':
        dataset = RGBD_NYU_dataset(rgb_path=rgb_path,
                                   dep_path=dep_path,
                                   lab_path=lab_path,
                                   mode='train',
                                   transform=transforms_depth())

        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=False,
                                drop_last=True)



    else:
        print('no {} depth type'.format(mode))
        os._exit(0)

    return dataset, dataloader


def warm_up(optimizer, warm_epoch, epoch, warm_up_lr, lr):
    # increase the low lr to normal lr gradually
    if epoch < warm_epoch:
        print('warm up in epoch {}'.format(epoch))
        optimizer.param_groups[0]['lr'] = warm_up_lr + ((lr - warm_up_lr) / (warm_epoch - 1)) * (epoch - 1)
    elif epoch == warm_epoch:
        print('warm up in epoch {}'.format(epoch))
        optimizer.param_groups[0]['lr'] = lr

    return optimizer

