import numpy as np
import torch
import torch.optim.lr_scheduler
from torchvision.utils import make_grid
import argparse
import math
from tensorboardX import SummaryWriter
from Utils.utils import predit2label, print_log, save_metrics, depth_mode, test_metrics, warm_up, color_label
from Utils.loss import CrossEntropyLoss2d
from Nets.RDFNet import RDFNet50_HHA
from Nets.CMAnet import CMAnet

from Utils.load import load_checkpoint, save_checkpoint
from Utils.metrics import SegmentationMetric
import time
import warnings

# igonre warnings, which causing print error
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="CMANet Indoor Semantic Segmentation")

parser.add_argument('--device', type=str, default='cuda', help='choose device')
parser.add_argument('--numclass', type=int, default=40, help='')
parser.add_argument('--learning_rate', type=float, default=1.2e-3, help='')
parser.add_argument('--warmup_lr', type=float, default=1e-4, help='')
parser.add_argument('--warmup_epoch', type=int, default=15, help='')
parser.add_argument('--epochs', type=int, default=800, help='')
parser.add_argument('--batch_size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--reload', default=False, type=bool,
                    help='')
parser.add_argument('--depth_type', default='hha', type=str, help='')

args = parser.parse_args()

rgbpath = r'datasets/NYU2/train_rgb'
hhapath = r'datasets/NYU2/train_hha'
deppath = r'datasets/NYU2/train_depth'
labpath = r'datasets/NYU2/train_class40'

image_h = 480
image_w = 640

load_dir = r'ckpt_epoch_350.00.pth'

summary_dir = r'savefile'


def train():
    # Using dataloader and dataset to read the image/depth/label
    device = args.device

    dataset, dataloader = depth_mode(mode=args.depth_type,
                                     batchsize=args.batch_size,
                                     rgb_path=rgbpath,
                                     dep_path=deppath,
                                     hha_path=hhapath,
                                     lab_path=labpath)

    # net = RDFNet50_HHA(num_classes=args.numclass, mode='train')
    net = CMAnet(num_classes=args.numclass)

    print('net')

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8, last_epoch=-1,
                                                verbose=False)

    # load and freeze resnet params
    net._load_resnet_pretrained()
    net._freeze_parameters()
    net._unfreeze_bn()

    # reload or not
    pre_epoch = 1
    if args.reload == True:
        pre_epoch = load_checkpoint(net, optimizer, scheduler, load_dir)
        net._unfreeze_all_parameters()
    # build tensorboard
    writer = SummaryWriter(summary_dir, comment="handw_exp", filename_suffix="CMANet")

    # the number of iterations in one epoch
    itr_epoch = math.floor(len(dataset) / args.batch_size)

    # set loss functions
    CEL_weighted = CrossEntropyLoss2d()

    # change device
    net.to(device)
    CEL_weighted.to(device)

    training_sum_time = 0
    print("begin training!")
    for epoch in range(pre_epoch, args.epochs + 1):
        # train mode
        net.train()

        # caculate time and metrics
        epoch_start_time = time.time()

        # metrics caculate
        train_metricclass = SegmentationMetric(numClass=args.numclass)

        # warm_up
        optimizer = warm_up(optimizer=optimizer,
                            warm_epoch=args.warmup_epoch,
                            epoch=epoch,
                            warm_up_lr=args.warmup_lr,
                            lr=args.learning_rate)

        # caculate epoch sum loss
        epoch_loss = 0

        for batch_idx, sample in enumerate(dataloader):
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
            optimizer.zero_grad()

            # get the result through net(input)
            pred_scales = net(image, depth)

            # loss caculation
            loss = CEL_weighted(pred_scales, target_scales)

            # update net parameters
            loss.backward()
            optimizer.step()

            # metric caculate
            train_metricclass.addBatch(imgPredict=predit2label(pred_scales[0].to('cpu')),
                                       imgLabel=target_scales[0].to('cpu') - 1)
            Pa = train_metricclass.pixelAccuracy()
            mPa = train_metricclass.meanPixelAccuracy()
            mIOU = train_metricclass.meanIntersectionOverUnion()

            # caculate time
            itr_end_time = time.time() - epoch_start_time

            # print and save
            print_log(epoch, optimizer.param_groups[0]['lr'], itr_end_time, batch_idx, itr_epoch, loss.item(),
                      Pa, mPa, mIOU)

            epoch_loss += loss.item()

        # save metrics after each epoch
        save_metrics(epoch, optimizer.param_groups[0]['lr'], itr_end_time, epoch_loss / itr_epoch, Pa, mPa, mIOU)
        training_sum_time += itr_end_time

        # update learning rate
        # scheduler.step(epoch_loss / itr_epoch)
        scheduler.step()
        # tensorboard save
        if epoch % 5 == 0 or epoch == 1:
            for name, param in net.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch, bins='doane')
            grid_image = make_grid(image[:4].clone().cpu().data, 4, normalize=True)
            writer.add_image('image', grid_image, epoch)
            grid_image = make_grid(depth[:4].clone().cpu().data, 4, normalize=True)
            writer.add_image('depth', grid_image, epoch)
            grid_image = make_grid(color_label(torch.max(pred_scales[0][:4], 1)[1] + 1), 4, normalize=False,
                                   range=(0, 255))
            writer.add_image('Predicted label1', grid_image, epoch)
            grid_image = make_grid(color_label(torch.max(pred_scales[1][:4], 1)[1] + 1), 4, normalize=False,
                                   range=(0, 255))
            writer.add_image('Predicted label2', grid_image, epoch)
            grid_image = make_grid(color_label(torch.max(pred_scales[2][:4], 1)[1] + 1), 4, normalize=False,
                                   range=(0, 255))
            writer.add_image('Predicted label3', grid_image, epoch)
            grid_image = make_grid(color_label(torch.max(pred_scales[3][:4], 1)[1] + 1), 4, normalize=False,
                                   range=(0, 255))
            writer.add_image('Predicted label4', grid_image, epoch)
            grid_image = make_grid(color_label(torch.max(pred_scales[4][:4], 1)[1] + 1), 4, normalize=False,
                                   range=(0, 255))
            writer.add_image('Predicted label5', grid_image, epoch)
            grid_image = make_grid(color_label(target_scales[0][:4]), 4, normalize=False, range=(0, 255))
            writer.add_image('Groundtruth label', grid_image, epoch)
            writer.add_scalar('CrossEntropyLoss', loss.data, global_step=epoch)
            writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=epoch)
            writer.add_scalar('Training Sum Time', np.around(training_sum_time/3600, decimals=2), global_step=epoch)
            writer.add_scalar('Itr Time', itr_end_time, global_step=epoch)
            writer.add_scalar('PixelAccuracy', Pa, global_step=epoch)
            writer.add_scalar('MeanPixelAccuracy', mPa, global_step=epoch)
            writer.add_scalar('MeanIOU', mIOU, global_step=epoch)
            print('tensorboard saved in epoch {}'.format(epoch))

        # save model each 25 epochs
        if epoch % 10 == 0:
            save_checkpoint(model=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            print('checkpoint saved in epoch {}'.format(epoch))

        # save test dataset metrics in each 50 epochs
        if epoch > 450 and epoch % 20 == 0:
            test_metrics(epoch=epoch, net=net, depth_type=args.depth_type, numclass=args.numclass)
            print('test metrics saved in epoch {}'.format(epoch))

        # unfreeze some layers in resnet
        if epoch == 60:
            net._unfreeze_layers_parameters()

        # unfreeze all parameters while epoch = 100 for encoder training
        if epoch == 80:
            net._unfreeze_all_parameters()

        # empty cuda
        torch.cuda.empty_cache()

    print("train completed!")


if __name__ == "__main__":
    train()
