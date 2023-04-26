import logging
import os
import sys
import json

import numpy as np
import torch

from monai import config
from monai.data import ArrayDataset, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity

from Model.EntireModel import EntireModel
from Model.TeacherModel import TeacherModel


def main(config):
    # config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images = []

    if not os.path.exists(config['data_dir']):
        raise FileNotFoundError

    for name in os.listdir(config['data_dir']):
        img_path = os.path.join(config['data_dir'], name)
        images.append(img_path)

    # define transforms for image and segmentation
    imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

    if config['is_calculate']:
        if not os.path.exists(config['gt_dir']):
            raise FileNotFoundError
        segments = []

        for name in os.listdir(config['gt_dir']):
            seg_path = os.path.join(config['gt_dir'], name)
            segments.append(seg_path)

        segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

    if not config['is_calculate']:
        img_ds = ArrayDataset(images, imtrans)
    else:
        img_ds = ArrayDataset(images, imtrans, segments, segtrans)

    data_loader = DataLoader(img_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    # sliding window inference for one image at every iteration
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=True, threshold=0.5)])

    if not os.path.exists(config['seg_dir']):
        os.mkdir(config['seg_dir'])
    saver = SaveImage(output_dir=config['seg_dir'], output_ext=".tif", output_postfix="seg", output_dtype=np.uint16,
                      writer="ITKWriter")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config['state_dir']):
        raise FileNotFoundError
    state = torch.load(config['state_dir'])
    entire_model = EntireModel(in_channel=config['in_channel'],
                               num_classes=config['num_classes']
                               )
    state['state_dict'] = {key[7:]: state['state_dict'][key] for key in state['state_dict']}
    entire_model.load_state_dict(state['state_dict'])
    model = TeacherModel(entire_model.encoder_t, entire_model.decoder_t).to(device)
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].permute(0, 1, 4, 2, 3).to(device), data[1].permute(0, 1, 4, 2, 3).to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (64, 64, 64)
            sw_batch_size = 4
            # print(images.shape)
            val_outputs = sliding_window_inference(images, roi_size, sw_batch_size, model)
            # print(val_outputs.shape)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            labels = decollate_batch(labels)
            dice_metric(y_pred=val_outputs, y=labels)
            # compute metric for current iteration
            for val_output in val_outputs:
                # print(val_output.shape)
                saver(val_output.permute(0, 2, 3, 1))
        print("Finish one image!")
        # aggregate the final mean dice result
        print("evaluation metric:", dice_metric.aggregate().item())
        # reset the status
        dice_metric.reset()


if __name__ == "__main__":
    config = json.load(open("./configs/config_predict.json"))
    main(config)
