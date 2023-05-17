import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.network_helpers import init_weights

feat_channels = [16, 32, 64, 128, 256]


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                 init_stride=(1, 1, 1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True), )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class Conv3D_Block(nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=False):

        super(Conv3D_Block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(inp_feat, out_feat, kernel_size=kernel,
                      stride=stride, padding=padding, bias=True),
            nn.InstanceNorm3d(out_feat),
            nn.PReLU())

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=kernel,
                      stride=stride, padding=padding, bias=True),
            nn.InstanceNorm3d(out_feat),
            nn.PReLU())

        self.residual = residual

        if self.residual:
            self.residual_upsampler = nn.Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = nn.Sequential(
            # 3D反卷积
            nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                               stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=0,
                               bias=True),
            nn.PReLU())

    def forward(self, x):
        return self.deconv(x)


class EncoderNetwork(nn.Module):
    def __init__(self, in_channel, is_batchnorm=True):
        super(EncoderNetwork, self).__init__()

        # downsampling
        self.conv1 = UnetConv3(in_channel, feat_channels[0], is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(feat_channels[0], feat_channels[1], is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(feat_channels[1], feat_channels[2], is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(feat_channels[2], feat_channels[3], is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(feat_channels[3], feat_channels[4], is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, data):
        f1 = self.conv1(data)
        maxpool1 = self.maxpool1(f1)

        f2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(f2)

        f3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(f3)

        f4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(f4)

        f5 = self.center(maxpool4)

        return f1, f2, f3, f4, f5


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def cal_d(d, f1, f2, f3, f4, f5, decoder, it=1, pred=None):
    for _ in range(it):
        pred_hat = decoder(f1, f2, f3, f4, f5)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    return d


def get_r_adv_t_single(d, f1, f2, f3, f4, f5, decoder, it=1, xi=1e-1, eps=10.0, pred=None):
    if pred is None:
        raise ValueError
    # f1, f2, f3, f4, f5 = f1.clone(), f2.clone(), f3.clone(), f4.clone(), f5.clone()
    if d.shape == f1.shape:
        d = cal_d(d, f1 + xi * d, f2, f3, f4, f5, decoder, it, pred)
    elif d.shape == f2.shape:
        d = cal_d(d, f1, f2 + xi * d, f3, f4, f5, decoder, it, pred)
    elif d.shape == f3.shape:
        d = cal_d(d, f1, f2, f3 + xi * d, f4, f5, decoder, it, pred)
    elif d.shape == f4.shape:
        d = cal_d(d, f1, f2, f3, f4 + xi * d, f5, decoder, it, pred)
    elif d.shape == f5.shape:
        d = cal_d(d, f1, f2, f3, f4, f5 + xi * d, decoder, it, pred)
    else:
        raise ValueError

    return d * eps


def get_r_adv_t_mul(f1, f2, f3, f4, f5, decoder, it=1, xi=1e-1, eps=10.0):
    decoder.eval()

    f1_de, f2_de, f3_de, f4_de, f5_de = f1.detach(), f2.detach(), f3.detach(), f4.detach(), f5.detach()

    with torch.no_grad():
        pred = F.softmax(decoder(f1_de, f2_de, f3_de, f4_de, f5_de), dim=1)

    d_1 = torch.rand(f1.shape).sub(0.5).to(f1.device)
    d_1 = _l2_normalize(d_1)
    d_1.requires_grad_()
    d_2 = torch.rand(f2.shape).sub(0.5).to(f2.device)
    d_2 = _l2_normalize(d_2)
    d_2.requires_grad_()
    d_3 = torch.rand(f3.shape).sub(0.5).to(f3.device)
    d_3 = _l2_normalize(d_3)
    d_3.requires_grad_()
    d_4 = torch.rand(f4.shape).sub(0.5).to(f4.device)
    d_4 = _l2_normalize(d_4)
    d_4.requires_grad_()
    d_5 = torch.rand(f5.shape).sub(0.5).to(f5.device)
    d_5 = _l2_normalize(d_5)
    d_5.requires_grad_()

    r_adv_1 = get_r_adv_t_single(d_1, f1_de, f2_de, f3_de, f4_de, f5_de, decoder, it, xi, eps, pred)
    r_adv_2 = get_r_adv_t_single(d_2, f1_de, f2_de, f3_de, f4_de, f5_de, decoder, it, xi, eps, pred)
    r_adv_3 = get_r_adv_t_single(d_3, f1_de, f2_de, f3_de, f4_de, f5_de, decoder, it, xi, eps, pred)
    r_adv_4 = get_r_adv_t_single(d_4, f1_de, f2_de, f3_de, f4_de, f5_de, decoder, it, xi, eps, pred)
    r_adv_5 = get_r_adv_t_single(d_5, f1_de, f2_de, f3_de, f4_de, f5_de, decoder, it, xi, eps, pred)

    decoder.train()
    return r_adv_1, r_adv_2, r_adv_3, r_adv_4, r_adv_5


class DecoderNetwork(nn.Module):
    def __init__(self, num_classes, is_batchnorm=True):
        super(DecoderNetwork, self).__init__()

        # upsampling
        self.up_concat4 = UnetUp3_CT(feat_channels[4], feat_channels[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(feat_channels[3], feat_channels[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(feat_channels[2], feat_channels[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(feat_channels[1], feat_channels[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(feat_channels[0], num_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, f1, f2, f3, f4, f5):

        center = self.dropout1(f5)
        up4 = self.up_concat4(f4, center)
        up3 = self.up_concat3(f3, up4)
        up2 = self.up_concat2(f2, up3)
        up1 = self.up_concat1(f1, up2)
        up1 = self.dropout2(up1)

        pred = self.final(up1)

        return pred


class TDecoderNetwork(nn.Module):
    def __init__(self, num_classes):
        super(TDecoderNetwork, self).__init__()
        self.decoder = DecoderNetwork(num_classes)

    def forward(self, f1, f2, f3, f4, f5):
        pred = self.decoder(f1, f2, f3, f4, f5)
        return pred


class VATDecoderNetwork(nn.Module):
    def __init__(self, num_classes):
        super(VATDecoderNetwork, self).__init__()
        self.decoder = DecoderNetwork(num_classes)

    def forward(self, f1, f2, f3, f4, f5, t_model=None, cur_w=.0):
        if t_model is not None:
            r_adv_1, r_adv_2, r_adv_3, r_adv_4, r_adv_5 = get_r_adv_t_mul(f1, f2, f3, f4, f5, t_model, it=1, xi=1e-6,
                                                                          eps=2.0)
            f1 = f1 + r_adv_1 * cur_w
            f2 = f2 + r_adv_2 * cur_w
            f3 = f3 + r_adv_3 * cur_w
            f4 = f4 + r_adv_4 * cur_w
            f5 = f5 + r_adv_5 * cur_w

        pred = self.decoder(f1, f2, f3, f4, f5)
        return pred
