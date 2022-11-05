import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from thop import profile

from tensorboardX import SummaryWriter

# Imagenet resnet
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                     kernel_size=3, padding=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes,
                               kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=4 * planes,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResidualConvUnit(nn.Module):
    def __init__(self, inplanes):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes))
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class ChainedResidualPool(nn.Module):
    def __init__(self, inplanes, blocks):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.blocks = blocks
        # one pool-block
        for i in range(0, self.blocks):
            self.add_module(
                "block{}".format(i + 1),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),  # obtain the raw feature map size
                    nn.Conv2d(inplanes, inplanes,
                              kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(inplanes)))

    def forward(self, x):
        x = self.relu(x)
        path = x
        for i in range(0, self.blocks):
            path = self.__getattr__("block{}".format(i + 1))(path)
            x = x + path
        return x


class CMChannelAttention(nn.Module):
    def __init__(self, channels_gate, reduction_ratio):
        super(CMChannelAttention, self).__init__()
        self.mlp_rgb1 = nn.Linear(2 * channels_gate, channels_gate // reduction_ratio, bias=False)
        self.mlp_rgb2 = nn.Linear(channels_gate // reduction_ratio, channels_gate, bias=False)
        self.mlp_dep1 = nn.Linear(2 * channels_gate, channels_gate // reduction_ratio, bias=False)
        self.mlp_dep2 = nn.Linear(channels_gate // reduction_ratio, channels_gate, bias=False)
        self.bn_rgb = nn.BatchNorm1d(channels_gate)
        self.bn_dep = nn.BatchNorm1d(channels_gate)

    def forward(self, rgb, dep):
        rgb_channel_max = F.max_pool2d(rgb, (rgb.size(2), rgb.size(3)), stride=(rgb.size(2), rgb.size(3)))
        rgb_channel_avg = F.avg_pool2d(rgb, (rgb.size(2), rgb.size(3)), stride=(rgb.size(2), rgb.size(3)))
        dep_channel_max = F.max_pool2d(dep, (dep.size(2), dep.size(3)), stride=(dep.size(2), dep.size(3)))
        dep_channel_avg = F.avg_pool2d(dep, (dep.size(2), dep.size(3)), stride=(dep.size(2), dep.size(3)))

        avg = torch.cat([rgb_channel_avg, dep_channel_avg], dim=1)
        max = torch.cat([rgb_channel_max, dep_channel_max], dim=1)

        rgb_avg_out1 = self.mlp_rgb1(torch.squeeze(torch.squeeze(avg, dim=2), dim=2))
        rgb_avg_out2 = self.mlp_rgb2(rgb_avg_out1)

        rgb_max_out1 = self.mlp_rgb1(torch.squeeze(torch.squeeze(max, dim=2), dim=2))
        rgb_max_out2 = self.mlp_rgb2(rgb_max_out1)

        dep_avg_out1 = self.mlp_dep1(torch.squeeze(torch.squeeze(avg, dim=2), dim=2))
        dep_avg_out2 = self.mlp_dep2(dep_avg_out1)

        dep_max_out1 = self.mlp_dep1(torch.squeeze(torch.squeeze(max, dim=2), dim=2))
        dep_max_out2 = self.mlp_dep2(dep_max_out1)

        rgb_scale = torch.sigmoid(self.bn_rgb(rgb_avg_out2 + rgb_max_out2)).unsqueeze(2).unsqueeze(3).expand_as(rgb)
        dep_scale = torch.sigmoid(self.bn_dep(dep_avg_out2 + dep_max_out2)).unsqueeze(2).unsqueeze(3).expand_as(dep)

        rgb_out = rgb * rgb_scale
        dep_out = dep * dep_scale

        return rgb_out, dep_out


class CMSpatialAttention(nn.Module):
    def __init__(self):
        super(CMSpatialAttention, self).__init__()
        self.conv2d_rgb = nn.Conv2d(in_channels=4, out_channels=1,
                                    kernel_size=7, stride=1, padding=3, bias=False)
        self.conv2d_dep = nn.Conv2d(in_channels=4, out_channels=1,
                                    kernel_size=7, stride=1, padding=3, bias=False)
        self.bn_rgb = nn.BatchNorm2d(1)
        self.bn_dep = nn.BatchNorm2d(1)

    def forward(self, rgb, dep):
        rgb_avg = torch.mean(rgb, dim=1, keepdim=True)
        rgb_max, _ = torch.max(rgb, dim=1, keepdim=True)
        dep_avg = torch.mean(dep, dim=1, keepdim=True)
        dep_max, _ = torch.max(dep, dim=1, keepdim=True)

        feature = torch.cat([rgb_avg, rgb_max, dep_avg, dep_max], dim=1)

        rgb_scale = torch.sigmoid(self.bn_rgb(self.conv2d_rgb(feature)))
        dep_scale = torch.sigmoid(self.bn_dep(self.conv2d_dep(feature)))

        rgb_out = rgb * rgb_scale
        dep_out = dep * dep_scale

        return rgb_out, dep_out


class CMCSAttention(nn.Module):
    def __init__(self, channels_gate, reduction_ratio):
        super(CMCSAttention, self).__init__()
        self.cmca = CMChannelAttention(channels_gate=channels_gate, reduction_ratio=reduction_ratio)
        self.cmsa = CMSpatialAttention()

    def forward(self, rgb, dep):
        rgb1, dep1 = self.cmca(rgb, dep)
        rgb2, dep2 = self.cmsa(rgb1, dep1)
        out = rgb2 + dep2
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels_gate, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.mlp1 = nn.Linear(channels_gate, channels_gate // reduction_ratio, bias=False)
        self.mlp2 = nn.Linear(channels_gate // reduction_ratio, channels_gate, bias=False)
        self.bn = nn.BatchNorm1d(channels_gate)

    def forward(self, input):
        channel_max_input = F.max_pool2d(input, (input.size(2), input.size(3)), stride=(input.size(2), input.size(3)))
        max_x1 = self.mlp1(torch.squeeze(torch.squeeze(channel_max_input, dim=2), dim=2))
        max_x2 = self.mlp2(max_x1)

        channel_avg_input = F.avg_pool2d(input, (input.size(2), input.size(3)), stride=(input.size(2), input.size(3)))
        avg_x1 = self.mlp1(torch.squeeze(torch.squeeze(channel_avg_input, dim=2), dim=2))
        avg_x2 = self.mlp2(avg_x1)

        scale = torch.sigmoid(self.bn(max_x2 + avg_x2)).unsqueeze(2).unsqueeze(3).expand_as(input)

        out = input * scale
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1,
                                kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, input):
        avgout = torch.mean(input, dim=1, keepdim=True)
        maxout, _ = torch.max(input, dim=1, keepdim=True)
        sa_map = self.bn(self.conv2d(torch.cat([avgout, maxout], dim=1)))
        scale = torch.sigmoid(sa_map)
        out = input * scale
        return out


class ConvBlockAttentionModule(nn.Module):
    def __init__(self, channels_gate, reduction_ratio):
        super(ConvBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(channels_gate=channels_gate, reduction_ratio=reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out + x


class BimodalContext(nn.Module):
    def __init__(self):
        super(BimodalContext, self).__init__()
        self.conv_rgb = nn.Conv2d(in_channels=2048, out_channels=512,
                                  kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_dep = nn.Conv2d(in_channels=2048, out_channels=512,
                                  kernel_size=1, stride=1, padding=0, bias=False)
        self.rcu_rgb = nn.Sequential(ResidualConvUnit(inplanes=512),
                                     ResidualConvUnit(inplanes=512),
                                     nn.Conv2d(in_channels=512, out_channels=512,
                                               kernel_size=3, padding=1, stride=1, bias=False))
        self.rcu_dep = nn.Sequential(ResidualConvUnit(inplanes=512),
                                     ResidualConvUnit(inplanes=512),
                                     nn.Conv2d(in_channels=512, out_channels=512,
                                               kernel_size=3, padding=1, stride=1, bias=False),
                                     )

        self.crp = ChainedResidualPool(inplanes=512, blocks=1)

        self.cbam = ConvBlockAttentionModule(channels_gate=512, reduction_ratio=4)

    def forward(self, rgb, dep):
        rgb1 = self.conv_rgb(rgb)
        dep1 = self.conv_dep(dep)
        rgb2 = self.rcu_rgb(rgb1)
        dep2 = self.rcu_dep(dep1)

        out = rgb2 + dep2
        out = self.crp(out)
        out = self.cbam(out)
        return out


class RefineAgent(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(RefineAgent, self).__init__()
        self.rgb_conv = nn.Sequential(nn.Conv2d(in_channels=inplanes, out_channels=outplanes,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(outplanes),
                                      ResidualConvUnit(outplanes)
                                      )
        self.dep_conv = nn.Sequential(nn.Conv2d(in_channels=inplanes, out_channels=outplanes,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(outplanes),
                                      ResidualConvUnit(outplanes)
                                      )

        self.crp = ChainedResidualPool(inplanes=outplanes, blocks=1)

        self.cbam = ConvBlockAttentionModule(channels_gate=outplanes, reduction_ratio=4)

    def forward(self, rgb, dep):
        rgb1 = self.rgb_conv(rgb)
        dep1 = self.dep_conv(dep)
        out = self.crp(rgb1 + dep1)
        out = self.cbam(out)
        return out


class CMAnet(nn.Module):
    def __init__(self, num_classes):
        super(CMAnet, self).__init__()
        self.backbone = 'resnet50'
        self.freezedict = {}

        block = Bottleneck
        transblock = TransBasicBlock
        layers = [3, 4, 6, 3]
        ca_ratio = 8

        # RGB encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # depth/HHA encoder
        self.conv1_d = nn.Conv2d(in_channels=3, out_channels=64,
                                 kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.inplanes = 64
        self.layer1_d = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        # cm attention gate, ca(channel attention), sa(spatial attention)
        self.cma1 = self._make_cma(64, ratio=1)
        self.cma2 = self._make_cma(256, ratio=4)
        self.cma3 = self._make_cma(512, ratio=ca_ratio)
        self.cma4 = self._make_cma(1024, ratio=ca_ratio)
        self.cma5 = self._make_cma(2048, ratio=ca_ratio)

        # refineagent
        self.agent1 = self._make_refineagent(inplanes=64, outplanes=64)
        self.agent2 = self._make_refineagent(inplanes=64 * 4, outplanes=64)
        self.agent3 = self._make_refineagent(inplanes=128 * 4, outplanes=128)
        self.agent4 = self._make_refineagent(inplanes=256 * 4, outplanes=256)

        # bimodal context
        self.bmcontext = self._make_bimodalcontext()

        # decoder stage
        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv5 = self._make_transpose(transblock, 64, 3, stride=1)

        # output stage
        self.out5_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.out4_conv = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=True)
        self.out3_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
        self.out1_conv = nn.ConvTranspose2d(64, num_classes, kernel_size=2,
                                            stride=2, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, rgb, dep):
        # stage1
        # rgb encoder1
        rgb1 = self.conv1(rgb)
        rgb1 = self.bn1(rgb1)
        rgb1 = self.relu(rgb1)  # (64,240,320)

        # dep encoder1
        dep1 = self.conv1_d(dep)
        dep1 = self.bn1_d(dep1)
        dep1 = self.relu(dep1)  # (64,240,320)

        # cma1
        cmf1 = self.cma1(rgb1, dep1)  # (64,240,320)
        # BMP2
        rgb11 = (rgb1 + cmf1) / 2
        dep11 = (dep1 + cmf1) / 2

        # refineagent1
        skipagent1 = self.agent1(rgb11, dep11)  # (64,240,320)

        rgb12 = self.maxpool(rgb11)  # (64,120,160)
        dep12 = self.maxpool(dep11)  # (64,120,160)

        # stage2
        # encoder2
        rgb2 = self.layer1(rgb12)  # (256,120,160)
        dep2 = self.layer1_d(dep12)  # (256,120,160)
        # cma2
        cmf2 = self.cma2(rgb2, dep2)  # (64,120,160)
        # BMP2
        rgb22 = (rgb2 + cmf2) / 2
        dep22 = (dep2 + cmf2) / 2
        # refineagent2
        skipagent2 = self.agent2(rgb22, dep22)  # (64,120,160)

        # stage3
        # encoder3
        rgb3 = self.layer2(rgb22)  # (512,60,80)
        dep3 = self.layer2_d(dep22)  # (512,60,80)
        # cma3
        cmf3 = self.cma3(rgb3, dep3)  # (128,60,80)
        # BMP3
        rgb32 = (rgb3 + cmf3) / 2
        dep32 = (dep3 + cmf3) / 2
        # refineagent3
        skipagent3 = self.agent3(rgb32, dep32)  # (128,60,80)

        # stage4
        # encoder4
        rgb4 = self.layer3(rgb32)  # (1024,30,40)
        dep4 = self.layer3_d(dep32)  # (1024,30,40)
        # cma4
        cmf4 = self.cma4(rgb4, dep4)  # (256,30,40)
        # BMP4
        rgb42 = (rgb4 + cmf4) / 2
        dep42 = (dep4 + cmf4) / 2
        # refineagent4
        skipagent4 = self.agent4(rgb42, dep42)  # (256,30,40)

        # stage5
        # encoder5
        rgb5 = self.layer4(rgb42)  # (2048,15,20)
        dep5 = self.layer4_d(dep42)  # (2048,15,20)
        # cma5
        cmf5 = self.cma5(rgb5, dep5)
        rgb52 = (rgb5 + cmf5) / 2
        dep52 = (dep5 + cmf5) / 2

        # stage6
        # bimodal context
        de_feat = self.bmcontext(rgb52, dep52)  # (512,15,20)

        # stage7
        # decoder1
        de1 = self.deconv1(de_feat)  # (256,30,40)
        # output5
        out5 = self.out5_conv(de1)  # (numclass,30,40)

        # stage8
        # decoder2
        de2 = de1 + skipagent4
        de22 = self.deconv2(de2)  # (128,60,80)
        # output4
        out4 = self.out4_conv(de22)  # (numclass,60,80)

        # stage9
        # decoder3
        de3 = de22 + skipagent3
        de32 = self.deconv3(de3)  # (64,120,160)
        # output3
        out3 = self.out3_conv(de32)  # (numclass,120,160)

        # stage10
        # decoder4
        de4 = de32 + skipagent2
        de42 = self.deconv4(de4)  # (64,240,320)
        # output2
        out2 = self.out2_conv(de42)  # (numclass,240,320)

        # stage11
        # decoder5
        de5 = de42 + skipagent1
        de52 = self.deconv5(de5)  # (64,240,320)
        # output1
        out1 = self.out1_conv(de52)  # (numclass,480,640)

        return [out1, out2, out3, out4, out5]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_cmca(self, inplanes, ratio):
        return CMChannelAttention(channels_gate=inplanes, reduction_ratio=ratio)

    def _make_cmsa(self):
        return CMSpatialAttention()

    def _make_cma(self, inplanes, ratio):
        return CMCSAttention(channels_gate=inplanes, reduction_ratio=ratio)

    def _make_bimodalcontext(self):
        return BimodalContext()

    def _make_refineagent(self, inplanes, outplanes):
        return RefineAgent(inplanes=inplanes, outplanes=outplanes)

    def _load_resnet_pretrained(self):
        print('loading {} params!'.format(self.backbone))
        pretrain_dict = model_zoo.load_url(model_urls[self.backbone])
        model_dict = {}
        state_dict = self.state_dict()
        for name, para in pretrain_dict.items():
            if name in state_dict:
                if name.startswith('conv1'):
                    model_dict[name] = para
                    model_dict[name.replace('conv1', 'conv1_d')] = para
                elif name.startswith('bn1'):
                    model_dict[name] = para
                    model_dict[name.replace('bn1', 'bn1_d')] = para
                elif name.startswith('layer'):
                    model_dict[name] = para
                    model_dict[name[:6] + '_d' + name[6:]] = para
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        print('{} params loaded!'.format(self.backbone))

        self.freezedict = model_dict

    def _freeze_parameters(self):
        # cant freeze if not load resnet, because the freeze_dict is null
        assert (bool(self.freezedict) == True), 'freeze params must after loaded ResNet!'
        print('freezing resnet paras!')
        freeze_dict = self.freezedict
        state_dict = self.state_dict(keep_vars=True)
        for k, v in freeze_dict.items():
            if k in state_dict:
                state_dict[k].requires_grad = False
        print('resnet paras freezed !')

    def _unfreeze_bn(self):
        # freeze parameters in ResNet except BatchNorm!!!!
        # or the batchnorm layer will not trainable, but keep the mean and var (weights and bias) on Imagenet dataset
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
        print('all bn unfreezed!')

    def _unfreeze_all_parameters(self):
        print('unfreezing all paras!')
        for name, para in self.named_parameters():
            para.requires_grad = True
        print('all paras unfreezed!')

    def _unfreeze_layers_parameters(self, layers=['conv1', 'conv1_d', 'layer1', 'layer1_d']):
        unfreeze_layer_res = layers
        print('unfreeze layers : \n {}'.format(unfreeze_layer_res))
        for layer in unfreeze_layer_res:
            for param in self.__getattr__(layer).parameters():
                param.requires_grad = True
        print('layers unfreezed !')


if __name__ == '__main__':
    net = CMAnet(num_classes=40)
    net._load_resnet_pretrained()
    net._freeze_parameters()
    net._unfreeze_bn()
    net._unfreeze_layers_parameters()
    net._unfreeze_all_parameters()
    # for name, para in net.state_dict(keep_vars=True).items():
    #     print(name, para.shape, para.requires_grad)

    rgb = torch.randn([1, 3, 480, 640])
    dep = torch.randn([1, 3, 480, 640])
    flops, params = profile(net, inputs=(rgb, dep))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))

    # writer = SummaryWriter(log_dir='./a', comment="handw_exp", filename_suffix="test1")
    # rgb = torch.randn([2, 3, 480, 640])
    # dep = torch.randn([2, 3, 480, 640])
    # writer.add_graph(model=net, input_to_model=(rgb, dep))
    # for i in range(3):
    #     for name, param in net.named_parameters():
    #         writer.add_histogram(name + '_data', param, i, bins='doane')
    #     print(i)
    import time

    depth = torch.randn([1, 3, 64, 64])
    rgbbb = torch.randn([1, 3, 64, 64])
    net.eval()
    start_time = time.time()

    net.to(device='cuda')
    depth = depth.to(device='cuda')
    rgbbb = rgbbb.to(device='cuda')
    output = net(rgbbb, depth)
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    print('inferencetime:{:.1f}'.format(total_time))
