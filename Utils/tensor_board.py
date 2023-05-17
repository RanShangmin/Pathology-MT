import os
import PIL
import wandb
import numpy
import torch
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt


class Tensorboard:
    def __init__(self, config, online=False):
        os.environ['WANDB_API_KEY'] = ""
        os.system("wandb login")
        os.system("wandb {}".format("online" if online else "offline"))
        self.tensor_board = wandb.init(project=config['name'], name=config['experim_name'],
                                       config=config)

        self.current_step = 0
        self.root_dir = os.path.join(config['trainer']['save_dir'],
                                     config['experim_name'])

    def step_forward(self, global_step):
        self.current_step = global_step

    def upload_single_info(self, info):
        key, value = info.popitem()
        self.tensor_board.log({key: value,
                               "global_step": self.current_step})
        return

    def upload_wandb_info(self, info_dict):
        for i, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info],
                                   "global_step": self.current_step})
        return

    def update_table(self, table_info, axis_name, title=""):
        x_name, y_name = axis_name['x'], axis_name['y']
        table = wandb.Table(data=table_info,
                            columns=[x_name, y_name])
        wandb.log({title: wandb.plot.bar(table, x_name, y_name, title=title)})

    @staticmethod
    def finish():
        wandb.finish()
