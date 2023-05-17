import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from skimage import transform
from monai.transforms.intensity.array import RandScaleIntensity, RandStdShiftIntensity, RandGaussianSmooth, \
    RandGaussianSharpen, RandAdjustContrast, RandHistogramShift, RandGaussianNoise, NormalizeIntensity
from Utils.param_trans import trans_data, trans_label


# Ensure your dataset only contains 3d gray images!
class BaseDataSet(Dataset):
    def __init__(self, data_dir, split, weak_times=1, base_size=None, reflect_index=None, augment=True, val=False,
                 use_weak_lables=False, weak_labels_output=None, crop_size=None, scale=False,
                 flip=False, rotate=False, n_labeled_examples=None):

        self.root = data_dir
        self.split = split
        self.augment = augment
        self.crop_size = crop_size
        self.n_labeled_examples = n_labeled_examples
        self.val = val
        self.reflect_index = reflect_index
        self.use_weak_lables = use_weak_lables
        self.weak_labels_output = weak_labels_output
        self.weak_times = weak_times
        self.base_size = base_size

        if self.augment:
            self.scale = scale
            self.flip = flip
            self.rotate = rotate

        self.files = []
        self._set_files()

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index):
        raise NotImplementedError

    def _rotation_3d(self, image, axis, theta, expand=False, fill=0.0, interpolation=InterpolationMode.BILINEAR):
        """
        The rotation is based on torchvision.transforms.functional.rotate, which is originally made for a 2d image rotation
        :param x: the data that should be rotated, a torch.tensor or an ndarray
        :param axis: the rotation axis based on the keynote request. 0 for x axis, 1 for y axis, and 2 for z axis.
        :param expand:  (bool, optional) – Optional expansion flag. If true, expands the output image to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
        :param fill:  (sequence or number, optional) –Pixel fill value for the area outside the transformed image. If given a number, the value is used for all bands respectively.
        :param theta: the rotation angle, Counter-clockwise rotation, [-180, 180] degrees.
        :return: rotated tensor.
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
            # X = X.float()

        image = image.to(device)

        if axis == 0:
            image = rotate(image, interpolation=interpolation, angle=theta, expand=expand, fill=fill)
        elif axis == 1:
            image = image.permute((1, 0, 2))
            image = rotate(image, interpolation=interpolation, angle=theta, expand=expand, fill=fill)
            image = image.permute((1, 0, 2))
        elif axis == 2:
            image = image.permute((2, 1, 0))
            image = rotate(image, interpolation=interpolation, angle=-theta, expand=expand, fill=fill)
            image = image.permute((2, 1, 0))
        else:
            raise Exception('Not invalid axis')
        return image.squeeze(0).cpu()

    def _rotate90(self, image, label):
        # Rotate the image with an angle between -90 and 90
        angle = random.randint(-1, 2)
        axis = random.randint(0, 2)
        image = self._rotation_3d(image, axis, angle * 90, fill=.0, interpolation=InterpolationMode.BILINEAR)
        label = self._rotation_3d(label, axis, angle * 90, fill=0, interpolation=InterpolationMode.NEAREST)
        return image, label

    def _flip(self, image, label):
        axis = random.randint(0, 2)
        if random.random() > 0.5:
            # image = torch.flip(image, dims=[axis]).clone()
            # label = torch.flip(label, dims=[axis]).clone()
            # print(image.dtype, label.dtype)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        return image, label

    def _crop(self, image, label):
        # Padding to return the correct crop size
        if (isinstance(self.crop_size, list) or isinstance(self.crop_size, tuple)) and len(self.crop_size) == 3:
            crop_d, crop_h, crop_w = self.crop_size
        elif isinstance(self.crop_size, int):
            crop_d, crop_h, crop_w = self.crop_size, self.crop_size, self.crop_size
        else:
            raise ValueError

        if crop_d == image.shape[0] and crop_h == image.shape[1] and crop_w == image.shape[2]:
            return image, label

        d, h, w = image.shape
        pad_d = max(crop_d - d, 0)
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)

        pad = np.zeros(6, np.integer)
        pad[0] = int(pad_d / 2)
        pad[1] = pad_d - pad[0]
        pad[2] = int(pad_h / 2)
        pad[3] = pad_h - pad[2]
        pad[4] = int(pad_w / 2)
        pad[5] = pad_w - pad[4]
        # pad_kwargs = {
        #     "top": 0,
        #     "bottom": pad_h,
        #     "left": 0,
        #     "right": pad_w,
        #     "borderType": cv2.BORDER_CONSTANT, }
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            # label = cv2.copyMakeBorder(label, value=self.ignore_index, **pad_kwargs)
            image = np.pad(image, (
                (pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5])),
                           constant_values=.0, mode='constant')
            label = np.pad(label, (
                (pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5])),
                           constant_values=0, mode='constant')

        # Cropping 
        d, h, w = image.shape
        start_d = random.randint(0, d - crop_d)
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        end_d = start_d + crop_d
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        image = image[start_d:end_d, start_h:end_h, start_w:end_w]
        label = label[start_d:end_d, start_h:end_h, start_w:end_w]
        return image, label

    def _resize(self, image, label, bigger_side_to_base_size=True):
        # return image, label
        if isinstance(self.base_size, int):
            d, h, w = image.shape
            if self.augment and self.scale:
                longside = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.2))
                # longside = random.randint(int(self.base_size*0.5), int(self.base_size*1))
            else:
                longside = self.base_size

            if bigger_side_to_base_size:
                # h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
                #     int(1.0 * longside * h / w + 0.5), longside)
                if d > h and d > w:
                    d, h, w = (longside, int(1.0 * longside * h / d + 0.5), int(1.0 * longside * w / d + 0.5))
                elif h > d and h > w:
                    d, h, w = (int(1.0 * longside * d / h + 0.5), longside, int(1.0 * longside * w / h + 0.5))
                else:
                    d, h, w = (int(1.0 * longside * d / w + 0.5), int(1.0 * longside * h / w + 0.5), longside)
            else:
                # h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h < w else (
                #     int(1.0 * longside * h / w + 0.5), longside)
                if d < h and d < w:
                    d, h, w = (longside, int(1.0 * longside * h / d + 0.5), int(1.0 * longside * w / d + 0.5))
                elif h < d and h < w:
                    d, h, w = (int(1.0 * longside * d / h + 0.5), longside, int(1.0 * longside * w / h + 0.5))
                else:
                    d, h, w = (int(1.0 * longside * d / w + 0.5), int(1.0 * longside * h / w + 0.5), longside)
            # image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
            image = transform.resize(image, (d, h, w), order=1, preserve_range=True)
            # label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            label = transform.resize(label, (d, h, w), order=0, preserve_range=True)
            return image, label

        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 3:
            if self.augment and self.scale:
                scale = random.random() * 0.4 + 0.8  # Scaling between [1, 1.5]
                d, h, w = int(self.base_size[0] * scale), int(self.base_size[1] * scale), int(self.base_size[2] * scale)
            else:
                d, h, w = self.base_size
            # image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
            image = transform.resize(image, (d, h, w), preserve_range=True)
            # label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            label = transform.resize(label, (d, h, w), order=0, preserve_range=True)
            return image, label

        else:
            raise ValueError

    def _repeat_weak(self, image):
        # # image = trans_data(image)
        # repeat_image = np.repeat(image, self.weak_times, axis=0)
        repeat_image = image.repeat(self.weak_times, 1, 1, 1)

        return repeat_image

    def _add_noise(self, image):
        add_noise = RandGaussianNoise(std=(torch.std(image).item() / 1))
        return add_noise(image)

    def _data_aug(self, image, flag="weak"):
        # print(image.dtype)
        weak_aug = self._repeat_weak(image)
        for i, img in enumerate(weak_aug):
            weak_aug[i] = self._add_noise(img)

        if flag == "weak":
            return weak_aug

        elif flag == "both":
            # kernel_size = int(random.random() * 4.95)
            # kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            # blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
            # shift_intensity = RandStdShiftIntensity(0.03)
            scale_intensity = RandScaleIntensity(0.5)
            shift_histogram = RandHistogramShift()
            smooth_image = RandGaussianSmooth()
            blurring_image = RandGaussianSharpen()
            color_jitter = RandAdjustContrast()

            strong_aug = image.clone()
            #
            # if random.random() < 0.1:
            #     strong_aug = shift_intensity(strong_aug)

            if random.random() < 0.2:
                strong_aug = self._add_noise(strong_aug)

            if random.random() < 0.2:
                strong_aug = scale_intensity(strong_aug)

            if random.random() < 0.2:
                strong_aug = shift_histogram(strong_aug)

            if random.random() < 0.2:
                strong_aug = color_jitter(strong_aug)

            if random.random() < 0.2:
                strong_aug = smooth_image(strong_aug)

            if random.random() < 0.2:
                # strong_aug = blurring_image(strong_aug)
                strong_aug = blurring_image(strong_aug)

            return weak_aug, strong_aug

        else:
            raise NotImplementedError

    def _normalize_intensity(self, image):
        if self.val or not self.augment or random.random() < 0.5:
            ni = NormalizeIntensity()
        else:
            ni = NormalizeIntensity(channel_wise=True)
        image = ni(image)
        # image /= 32.
        # print("mean:{},std:{}".format(np.mean(image), np.std(image)))
        return image

    def _augmentation(self, image, label):

        # image = Image.fromarray(np.float32(image))
        # image = self.jitter_tf(image) if self.jitter else image

        if self.flip:
            image, label = self._flip(image, label)

        if self.rotate:
            image, label = self._rotate90(image, label)

        # return self.normalize(self.to_tensor(image)), label
        image_wk, image_str = self._data_aug(image, flag="both")
        return image_wk, image_str, label

    def _reflect_index(self, label):
        if self.reflect_index is None:
            return label
        for index in range(len(self.reflect_index)):
            label[label == self.reflect_index[index]] = index
        # print(np.max(label))
        return label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)

        # print("image_id:{},mean:{},std:{}".format(image_id, np.mean(image), np.std(image)))

        if self.base_size is not None:
            image, label = self._resize(image, label)

        if self.crop_size is not None:
            image, label = self._crop(image, label)
        elif not self.val:
            raise ValueError

        image = self._normalize_intensity(image)
        label = self._reflect_index(label)

        label = np.array(label, dtype=np.int64)
        # label = torch.from_numpy(label).long()

        if self.val:
            return trans_data(image), trans_label(label)
        elif self.augment:
            image_wk, image_str, label = self._augmentation(image, label)
            return trans_data(image_wk), trans_data(image_str), trans_label(label)
        else:
            return trans_data(self._repeat_weak(image)), trans_data(image), trans_label(label)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
