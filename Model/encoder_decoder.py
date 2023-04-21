import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

feat_channels = [32, 64, 128, 256, 512]


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
    def __init__(self, in_channel, residual=True):
        super(EncoderNetwork, self).__init__()
        # Encoder downsamplers
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.pool4 = nn.MaxPool3d((2, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(in_channel, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

    def forward(self, data):
        f1 = self.conv_blk1(data)
        f_low1 = self.pool1(f1)
        f2 = self.conv_blk2(f_low1)
        f_low2 = self.pool2(f2)
        f3 = self.conv_blk3(f_low2)
        f_low3 = self.pool3(f3)
        f4 = self.conv_blk4(f_low3)
        f_low4 = self.pool4(f4)
        f5 = self.conv_blk5(f_low4)

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
    def __init__(self, num_classes, residual=True):
        super(DecoderNetwork, self).__init__()
        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = nn.Conv3d(feat_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, f1, f2, f3, f4, f5):
        # 解码器
        d4 = torch.cat([self.deconv_blk4(f5), f4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)
        d3 = torch.cat([self.deconv_blk3(d_high4), f3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d2 = torch.cat([self.deconv_blk2(d_high3), f2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d1 = torch.cat([self.deconv_blk1(d_high2), f1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        pred = self.one_conv(d_high1)
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

    def forward(self, f1, f2, f3, f4, f5, t_model=None):
        if t_model is not None:
            r_adv_1, r_adv_2, r_adv_3, r_adv_4, r_adv_5 = get_r_adv_t_mul(f1, f2, f3, f4, f5, t_model, it=1, xi=1e-6,
                                                                          eps=2.0)
            f1 = f1 + r_adv_1
            f2 = f2 + r_adv_2
            f3 = f3 + r_adv_3
            f4 = f4 + r_adv_4
            f5 = f5 + r_adv_5

        pred = self.decoder(f1, f2, f3, f4, f5)
        return pred
