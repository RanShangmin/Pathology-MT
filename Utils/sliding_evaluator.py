import torch
import numpy
import cv2
import collections


class SlidingEval(torch.nn.Module):
    def __init__(self, model, crop_size, stride_rate, device, class_number=2):
        super(SlidingEval, self).__init__()
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.device = device
        self.class_number = class_number
        self.model = model

    def forward(self, img):
        img = img.squeeze().cpu().numpy()
        assert len(img.shape) == 4 or len(img.shape) == 3
        if (len(img.shape) == 3):
            img = img[None, ...]
        _, ori_dep, ori_rows, ori_cols = img.shape
        processed_pred = numpy.zeros((1, self.class_number, ori_dep, ori_rows, ori_cols))
        processed_pred += self.scale_process(img, (ori_dep, ori_rows, ori_cols), self.device)
        return processed_pred

    # The img is normalized;
    def process_image(self, img, crop_size=None):
        p_img = img

        if crop_size is not None:
            p_img, margin = self.pad_image_to_shape(p_img, crop_size,
                                                    'constant', value=0)
            return p_img, margin

        return p_img

    def get_3dshape(self, shape, *, zero=True):
        if not isinstance(shape, collections.Iterable):
            shape = int(shape)
            shape = (shape, shape, shape)
        else:
            d, h, w = map(int, shape)
            shape = (d, h, w)
        if zero:
            minv = 0
        else:
            minv = 1

        assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
        return shape

    def pad_image_to_shape(self, img, shape, border_mode, value):
        margin = numpy.zeros(6, numpy.integer)
        shape = self.get_3dshape(shape)
        pad_depth = shape[0] - img.shape[1] if shape[0] - img.shape[1] > 0 else 0
        pad_height = shape[1] - img.shape[2] if shape[1] - img.shape[2] > 0 else 0
        pad_width = shape[2] - img.shape[3] if shape[2] - img.shape[3] > 0 else 0

        margin[0] = pad_depth // 2
        margin[1] = pad_depth // 2 + pad_depth % 2
        margin[2] = pad_height // 2
        margin[3] = pad_height // 2 + pad_height % 2
        margin[4] = pad_width // 2
        margin[5] = pad_width // 2 + pad_width % 2

        # img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
        #                          border_mode, value=value)
        img = numpy.pad(img, ((0, 0), (margin[0], margin[1]), (margin[2], margin[3]), (margin[4], margin[5])),
                        constant_values=value, mode=border_mode)

        return img, margin

    def scale_process(self, img, ori_shape, device=None):
        _, new_dep, new_rows, new_cols = img.shape
        if new_dep > new_rows and new_dep > new_cols:
            long_size = new_dep
        elif new_rows > new_dep and new_rows > new_cols:
            long_size = new_rows
        else:
            long_size = new_cols
        if isinstance(self.crop_size, int):
            self.crop_size = (self.crop_size, self.crop_size, self.crop_size)

        if long_size <= min(min(self.crop_size[0], self.crop_size[1]), self.crop_size[2]):
            input_data, margin = self.process_image(img, self.crop_size)  # pad image
            with torch.no_grad():
                input_data = torch.tensor(input_data, dtype=torch.float).cuda().unsqueeze(0)
                f1, f2, f3, f4 = self.model.module.encoder_t(input_data)
                score = self.model.module.decoder_t(f1, f2, f3, f4)
            score = score[:, :, margin[0]:(score.shape[2] - margin[1]),
                    margin[2]:(score.shape[3] - margin[3]),
                    margin[4]:(score.shape[4] - margin[5])]
        else:
            stride_0 = int(numpy.ceil(self.crop_size[0] * self.stride_rate))
            stride_1 = int(numpy.ceil(self.crop_size[1] * self.stride_rate))
            stride_2 = int(numpy.ceil(self.crop_size[2] * self.stride_rate))
            img_pad, margin = self.pad_image_to_shape(img, self.crop_size,
                                                      'constant', value=0)
            pad_dep = img_pad.shape[1]
            pad_rows = img_pad.shape[2]
            pad_cols = img_pad.shape[3]
            d_grid = int(numpy.ceil((pad_dep - self.crop_size[0]) / stride_0)) + 1
            r_grid = int(numpy.ceil((pad_rows - self.crop_size[1]) / stride_1)) + 1
            c_grid = int(numpy.ceil((pad_cols - self.crop_size[2]) / stride_2)) + 1
            data_scale = torch.zeros(1, self.class_number, pad_dep, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(1, self.class_number, pad_dep, pad_rows, pad_cols).cuda(
                device)
            for grid_zidx in range(d_grid):
                for grid_yidx in range(c_grid):
                    for grid_xidx in range(r_grid):
                        s_z = grid_zidx * stride_0
                        s_x = grid_xidx * stride_1
                        s_y = grid_yidx * stride_2
                        e_z = min(s_z + self.crop_size[0], pad_dep)
                        e_x = min(s_x + self.crop_size[1], pad_rows)
                        e_y = min(s_y + self.crop_size[2], pad_cols)
                        s_z = e_z - self.crop_size[0]
                        s_x = e_x - self.crop_size[1]
                        s_y = e_y - self.crop_size[2]
                        img_sub = img_pad[:, s_z:e_z, s_x:e_x, s_y:e_y]
                        count_scale[:, :, s_z:e_z, s_x:e_x, s_y:e_y] += 1
                        input_data, tmargin = self.process_image(img_sub, self.crop_size)
                        input_data = torch.tensor(input_data, dtype=torch.float).cuda().unsqueeze(0)
                        with torch.no_grad():
                            f1, f2, f3, f4 = self.model.module.encoder_t(input_data)
                            temp_score = self.model.module.decoder_t(f1, f2, f3, f4)
                        temp_score = temp_score[:, :,
                                     tmargin[0]:(temp_score.shape[2] - tmargin[1]),
                                     tmargin[2]:(temp_score.shape[3] - tmargin[3]),
                                     tmargin[4]:(temp_score.shape[4] - tmargin[5])]
                        data_scale[:, :, s_z:e_z, s_x:e_x, s_y:e_y] += temp_score

            # assert count_scale.min() > 0
            score = data_scale / count_scale
            score = score[:, :,
                    margin[0]:(score.shape[2] - margin[1]),
                    margin[2]:(score.shape[3] - margin[3]),
                    margin[4]:(score.shape[4] - margin[5])]

        # data_output = cv2.resize(score.cpu().numpy(),
        #                          (ori_shape[1], ori_shape[0]),
        #                          interpolation=cv2.INTER_LINEAR)
        data_output = score.cpu().numpy()
        # print(data_output.shape)

        return data_output
