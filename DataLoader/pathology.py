from Base.base_dataset import BaseDataSet
from Base.base_dataloader import BaseDataLoader

import numpy as np
import tifffile as tiff
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class PathologyDataset(BaseDataSet):
    def __init__(self, ddp_training, dgx, **kwargs):
        self.num_classes = 2
        self.ddp_training = ddp_training
        self.dgx = dgx
        # self.palette = get_voc_pallete(self.num_classes)
        super(PathologyDataset, self).__init__(**kwargs)

    def _set_files(self):
        # self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')
        prefix = "."
        if self.split == "val_10":
            file_list = os.path.join(prefix, "DataLoader/pathology_splits", f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(prefix, "DataLoader/pathology_splits", f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        # image = np.asarray(Image.open(image_path), dtype=np.float32)
        image = np.asarray(tiff.imread(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id + ".tif")
        else:
            label_path = os.path.join(self.root, self.labels[index][1:])
        # label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = np.asarray(tiff.imread(label_path), dtype=np.int32)
        return image, label, image_id


class Pathology(BaseDataLoader):
    def __init__(self, kwargs, ddp_training=False, dgx=False):
        self.MEAN = 128
        self.STD = 256
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')
        self.dataset = PathologyDataset(ddp_training, **kwargs, dgx=dgx)
        if ddp_training:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            train_sampler = None
        super(Pathology, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                   sampler=train_sampler)
