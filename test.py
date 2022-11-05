import torch
import torch.optim.lr_scheduler
import argparse
from Utils.utils import test_metrics
import warnings
from Nets.CMAnet import CMAnet

# igonre warnings, which causing print error
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="MFRNet Indoor Semantic Segmentation Test")

parser.add_argument('--device', type=str, default='cuda', help='choose device')
parser.add_argument('--numclass', type=int, default=40, help='')
parser.add_argument('--depth_type', default='hha', type=str, help='')

args = parser.parse_args()

lod_dir = 'ckpt_epoch_500.00.pth'

test_rgbpath = r'datasets/NYU2/test_rgb'
test_hhapath = r'datasets/NYU2/test_hha'
test_deppath = r'datasets/NYU2/test_depth'
test_labpath = r'datasets/NYU2/test_class40'

def test():
    device = args.device
    net = CMAnet(num_classes=args.numclass)
    net.to(device=device)
    net.train()
    print("=> loading checkpoint '{}'".format(lod_dir))
    checkpoint = torch.load(lod_dir, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'])
    print('{:>2} has been successfully loaded'.format(lod_dir))
    test_metrics(epoch=lod_dir, net=net, depth_type='hha', numclass=args.numclass)


if __name__ == '__main__':
    test()

