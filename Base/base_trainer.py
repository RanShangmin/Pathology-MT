import os
import json
import torch
from Utils import helpers
import Utils.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from itertools import chain
from Utils.helpers import group_weight


# from utils.htmlwriter import HTML


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class BaseTrainer:
    def __init__(self, model, config, iters_per_epoch, train_logger=None, args=None):
        self.model = model
        self.config = config
        self.args = args
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        # SETTING THE DEVICE
        if self.args.local_rank <= 0:
            self.logger = train_logger
            self.logger.critical(
                "distributed data parallel training: {}".format(str("on" if args.ddp is True
                                                                    else "off")))
        if self.args.ddp:
            torch.cuda.set_device(self.args.local_rank)
            self.model.cuda(self.args.local_rank)
            # if self.args.architecture == 'unet':
            #     init_weight(model.encoder_t.business_layer, torch.nn.init.kaiming_normal_,
            #                 torch.nn.BatchNorm2d, bn_eps, bn_momentum,
            #                 mode='fan_in', nonlinearity='relu')
            #     init_weight(model.decoder_t.business_layer, torch.nn.init.kaiming_normal_,
            #                 torch.nn.BatchNorm2d, bn_eps, bn_momentum,
            #                 mode='fan_in', nonlinearity='relu')

            # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            self.model = DDP(self.model, device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            # SETTING THE DEVICE
            self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
            self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        if args.architecture == "unet":
            params_list_t = []
            params_list_t = group_weight(params_list_t, model.encoder_t, config['optimizer']['args']['lr'])

            params_list_t = group_weight(params_list_t, model.decoder_t, config['optimizer']['args']['lr'])

            params_list_s = []
            params_list_s = group_weight(params_list_s, model.encoder_s, config['optimizer']['args']['lr'])

            params_list_s = group_weight(params_list_s, model.decoder_s, config['optimizer']['args']['lr'])

            self.optimizer_t = get_instance(torch.optim, 'optimizer', config, params_list_t)
            self.optimizer_s = get_instance(torch.optim, 'optimizer', config, params_list_s)

        else:
            raise NotImplementedError

        self.lr_scheduler_s = getattr(lr_scheduler, config['lr_scheduler'])(optimizer=self.optimizer_s,
                                                                            num_epochs=self.epochs,
                                                                            iters_per_epoch=iters_per_epoch)

        self.warm_up_epoch = config['model']['warm_up_epoch']
        # MONITORING
        self.mnt_current = .0

        # CHECKPOINTS
        run_name = config['experim_name']
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], run_name)
        if self.args.local_rank <= 0:
            helpers.dir_exists(self.checkpoint_dir)
            config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
            with open(config_save_path, 'w') as handle:
                json.dump(self.config, handle, indent=4, sort_keys=True)
        if self.args.ddp:
            dist.barrier()

    def train(self):
        if self.start_epoch <= self.warm_up_epoch:
            for epoch in range(0, self.warm_up_epoch):
                _ = self._warm_up(epoch, id=1)
                _ = self._warm_up(epoch, id=2)
                if epoch == self.warm_up_epoch - 1:
                    del self.optimizer_t
        self.model.module.freeze_teachers_parameters()
        for epoch in range(self.start_epoch, self.epochs + 1):
            _ = self._train_epoch(epoch, id=1)
            if self.args.ddp:
                dist.barrier()

            if epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(max(0, epoch))
                if self.args.local_rank <= 0:
                    self.logger.info('\n\n')
                    for k, v in results.items():
                        self.logger.info(f'         {str(k):15s}: {v}')
                    self.mnt_current = results["Dice"]
            else:
                continue

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0 and self.args.local_rank <= 0:
                self._save_checkpoint(epoch)
            if self.args.ddp:
                dist.barrier()

    def _save_checkpoint(self, epoch, name=""):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.mnt_current,
            'args': self.args
        }
        ckpt_name = str(self.args.labeled_examples) + '_Dice_{}_model{}_e{}.pth'.format(str(state['monitor_best']),
                                                                                        str(name), str(epoch))
        filename = os.path.join(self.checkpoint_dir, ckpt_name)
        self.logger.info('\nSaving a checkpoint: {} ...'.format(str(filename)))
        torch.save(state, filename)
        """
        pvc_dir = os.path.join("yy", "exercise_1", self.args.architecture,
                               "resnet{}_ckpt".format(str(self.args.backbone)), "city_cvpr_final",
                                                      str(self.args.labeled_examples))

        upload_checkpoint(local_path=self.checkpoint_dir, prefix=pvc_dir, checkpoint_filepath=ckpt_name)
        self.logger.info("Uploading current ckpt: mIoU_{}_model.pth to {}".format(str(state['monitor_best']), 
                                                                                  pvc_dir))
        """

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu

        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def _train_epoch(self, epoch, id):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError

    def _warm_up(self, epoch, id):
        raise NotImplementedError
