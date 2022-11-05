import warnings
from Nets.CMAnet_attentionmaps import CMAnet_attentionout
import argparse
import torch
import torch.nn.functional as F
import imageio
import cv2
import skimage.transform
import torchvision
import torch.optim
import os
from Utils.utils import predit2label, print_log, save_metrics, depth_mode, test_metrics, warm_up, color_label
import warnings
import numpy as np
from visualize import visualize_grid_attention_v2

# igonre warnings, which causing print error
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="MFRNet Indoor Semantic Segmentation Test")

parser.add_argument('--device', type=str, default='cuda', help='choose device')
parser.add_argument('--numclass', type=int, default=40, help='')

args = parser.parse_args()

lod_dir = r'ckpt_epoch_500.00.pth'

image_h = 480
image_w = 640

rgb_name = r'datasets\NYUV2\test_rgb\1250.png'
dep_name = r'datasets\NYUV2\test_hha\1250.png'


def visualization():
    device = args.device
    net = CMAnet_attentionout(num_classes=args.numclass)
    net.to(device=device)
    net.eval()
    print("=> loading checkpoint '{}'".format(lod_dir))
    checkpoint = torch.load(lod_dir, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'])
    print('{:>2} has been successfully loaded'.format(lod_dir))

    image = imageio.imread(rgb_name)
    depth = imageio.imread(dep_name)

    # Bi-linear
    image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                     mode='reflect', preserve_range=True)
    # Nearest-neighbor
    depth = skimage.transform.resize(depth, (image_h, image_w), order=1,
                                     mode='reflect', preserve_range=True)

    image = torch.from_numpy(image).float()
    depth = torch.from_numpy(depth).float()

    image = image / 255
    depth = depth / 255

    image = image.permute(2, 0, 1)
    depth = depth.permute(2, 0, 1)

    image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(image)
    depth = torchvision.transforms.Normalize(mean=[0.538, 0.444, 0.439],
                                             std=[0.211, 0.254, 0.146])(depth)

    image = image.to(device).unsqueeze_(0)
    depth = depth.to(device).unsqueeze_(0)

    with torch.no_grad():
        attentionmaps = net(image, depth)
        maps_dir = r'C:\Users\Administrator\Desktop\Attentionmaps'
        test_map = attentionmaps[3]

        # channel visualization
        for channels in range(test_map.size()[1]):
            maps = test_map[0, channels, :, :].unsqueeze(0)
            output = maps.cpu().numpy().transpose((1, 2, 0))
            output = skimage.transform.resize(output, (image_h, image_w), order=1,
                                              mode='reflect', preserve_range=True)
            output = output * 255
            imageio.imsave(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1)), output)

            # im_gray = cv2.imread(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1)), cv2.IMREAD_GRAYSCALE)
            # im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
            # cv2.imwrite(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1)), im_color)
            #
            img_rgb = skimage.io.imread(rgb_name)
            img_dep = skimage.io.imread(dep_name)
            img = img_rgb
            amap = cv2.cvtColor(skimage.io.imread(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1))), cv2.COLOR_RGB2BGR)

            normed_mask = amap / np.max(amap)
            normed_mask = np.uint8(255 * normed_mask)
            normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_BONE)
            # normed_mask = cv2.GaussianBlur(normed_mask, (101, 101), 100)
            normed_mask = cv2.addWeighted(img, 0.5, normed_mask, 2.0, 0)
            skimage.io.imsave(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1)), cv2.cvtColor(normed_mask, cv2.COLOR_BGR2RGB))

        # # spatial visualization
        # for channels in range(test_map.size()[1]):
        #     maps = test_map[0, channels, :, :].unsqueeze(0)
        #     output = maps.cpu().numpy().transpose((1, 2, 0))
        #     output = skimage.transform.resize(output, (image_h, image_w), order=1,
        #                                       mode='reflect', preserve_range=True)
        #     output= output * 255
        #     imageio.imsave(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1)), output)
        #
        #     # im_gray = cv2.imread(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1)), cv2.IMREAD_GRAYSCALE)
        #     # im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        #     # cv2.imwrite(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1)), im_color)
        #     #
        #     # img_rgb = skimage.io.imread(rgb_name)
        #     # img_dep = skimage.io.imread(dep_name)
        #     # img = img_rgb
        #     # amap = cv2.cvtColor(skimage.io.imread(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1))), cv2.COLOR_RGB2BGR)
        #     #
        #     # normed_mask = amap / np.max(amap)
        #     # # normed_mask = cv2.GaussianBlur(normed_mask, (101, 101), 100)
        #     # normed_mask = np.uint8(255 * normed_mask)
        #     # normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_AUTUMN)
        #     # normed_mask = cv2.addWeighted(img, 1.0, normed_mask, 1.0, 0)
        #     # skimage.io.imsave(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1)), cv2.cvtColor(normed_mask, cv2.COLOR_BGR2RGB))


        # normed_mask = amap / np.max(amap)
        # # normed_mask = cv2.GaussianBlur(normed_mask, (101, 101), 100)
        # normed_mask = np.uint8(255 * normed_mask)
        # normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_AUTUMN)
        # normed_mask = cv2.addWeighted(img, 1.0, normed_mask, 1.0, 0)
        # skimage.io.imsave(os.path.join(maps_dir, 'attentionmap{}.png'.format(channels+1)), cv2.cvtColor(normed_mask, cv2.COLOR_BGR2RGB))


        # GT = color_label(label)[0].numpy().transpose((1, 2, 0))
        # imageio.imsave(os.path.join(save_dir, 'label{}_GT.png'.format(rgb_dir[item][:-4])), GT)
        # for scale in range(len(pred)):
        #     output = F.interpolate(pred[scale], [image_h, image_w])
        #     output = color_label(torch.max(output, 1)[1] + 1)[0]
        #     output = output.cpu().numpy().transpose((1, 2, 0))
        #     imageio.imsave(os.path.join(save_dir, 'label{}_{}.png'.format(rgb_dir[item][:-4], scale + 1)), output)


if __name__ == '__main__':
    visualization()
