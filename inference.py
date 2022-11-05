from Nets.CMAnet import CMAnet
import argparse
import torch
import torch.nn.functional as F
import imageio
import skimage.transform
import torchvision
import torch.optim
import os
from Utils.utils import predit2label, print_log, save_metrics, depth_mode, test_metrics, warm_up, color_label
import warnings

# igonre warnings, which causing print error
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="MFRNet Indoor Semantic Segmentation Test")

parser.add_argument('--device', type=str, default='cuda', help='choose device')
parser.add_argument('--numclass', type=int, default=40, help='')

args = parser.parse_args()

lod_dir = r'model.pth'



test_rgbpath = r'Datasets\NYUV2\test_rgb'
test_hhapath = r'Datasets\NYUV2\test_hha'
test_deppath = r'Datasets\NYUV2\test_depth'
test_labpath = r'Datasets\NYUV2\test_class40'


save_dir = r'Save'
# save_dir = r'D:\Sci_study\Code\Python\CMANet\save'

image_h = 480
image_w = 640

def inference():
    device = args.device
    net = CMAnet(num_classes=args.numclass)
    net.to(device=device)
    net.eval()
    print("=> loading checkpoint '{}'".format(lod_dir))
    checkpoint = torch.load(lod_dir, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'])
    print('{:>2} has been successfully loaded'.format(lod_dir))

    rgb_dir = os.listdir(test_rgbpath)
    dep_dir = os.listdir(test_hhapath)
    lab_dir = os.listdir(test_labpath)

    for item in range(len(rgb_dir)):
        rgb_name = os.path.join(test_rgbpath, rgb_dir[item])
        dep_name = os.path.join(test_hhapath, dep_dir[item])
        lab_name = os.path.join(test_labpath, lab_dir[item])

        image = imageio.imread(rgb_name)
        depth = imageio.imread(dep_name)
        label = imageio.imread(lab_name)

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)


        image = torch.from_numpy(image).float()
        depth = torch.from_numpy(depth).float()
        label = torch.from_numpy(label/1.0).float()

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
        label = label.to(device).unsqueeze_(0)

        with torch.no_grad():
            pred = net(image, depth)
            GT = color_label(label)[0].numpy().transpose((1, 2, 0))
            imageio.imsave(os.path.join(save_dir, 'label{}_GT.png'.format(rgb_dir[item][:-4])), GT)
            for scale in range(1):
                output = pred[scale]
                output = F.interpolate(output, [image_h, image_w])
                output = color_label(torch.max(output, 1)[1] + 1)[0]
                output = output.cpu().numpy().transpose((1, 2, 0))
                imageio.imsave(os.path.join(save_dir,'label{}_{}.png'.format(rgb_dir[item][:-4], scale+1)), output)
            print('predit {} saved!'.format(rgb_dir[item]))



if __name__ == '__main__':
    inference()