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
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity

from Model.EntireModel import EntireModel
from Model.TeacherModel import TeacherModel


def main(config):
    # config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images = []

    if not os.path.exists(config['data_dir']):
        raise FileExistsError

    for name in os.listdir(config['data_dir']):
        img_path = os.path.join(config['data_dir'], name)
        images.append(img_path)

    # define transforms for image and segmentation
    imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    img_ds = ArrayDataset(images, imtrans)
    img_loader = DataLoader(img_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    # sliding window inference for one image at every iteration
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    if not os.path.exists(config['seg_dir']):
        os.mkdir(config['seg_dir'])
    saver = SaveImage(output_dir=config['seg_dir'], output_ext=".tif", output_postfix="seg", output_dtype=np.uint16,
                      writer="ITKWriter")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config['state_dir']):
        raise FileExistsError
    state = torch.load(config['state_dir'])
    entire_model = EntireModel(in_channel=config['in_channel'],
                               num_classes=config['num_classes']
                               )
    state['state_dict'] = {key[7:]: state['state_dict'][key] for key in state['state_dict']}
    entire_model.load_state_dict(state['state_dict'])
    model = TeacherModel(entire_model.encoder_t, entire_model.decoder_t).to(device)
    model.eval()
    with torch.no_grad():
        for img in img_loader:
            images = img.to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (64, 64, 64)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            # compute metric for current iteration
            for val_output in val_outputs:
                saver(val_output)
        print("Finish one image!")


if __name__ == "__main__":
    config = json.load(open("./configs/config_predict.json"))
    main(config)
