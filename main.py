import os
import random
import warnings
import argparse
import math
from train import Trainer
from Utils.losses import *
from DataLoader.pathology import Pathology
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from Utils.tensor_board import Tensorboard
from Model.EntireModel import EntireModel as model_deep

from Utils.logger import *

warnings.filterwarnings("ignore")


def main(gpu, ngpus_per_node, config, args):
    args.local_rank = gpu
    if args.local_rank <= 0:
        logger = logging.getLogger("DW-MT")
        logger.propagate = False
        logger.warning("Training start, total {} epochs".format(str(config['trainer']['epochs'])))
        logger.critical("GPU: {}".format(args.gpus))
        logger.critical("Network Architecture: {}".format(args.architecture))
        logger.critical("Current Labeled Example: {}".format(config['n_labeled_examples']))
        logger.critical(
            "Learning rate: other {} [world]".format(config['optimizer']['args']['lr']))

        logger.info("Image: {}x{} based on {}x{}".format(config['train_supervised']['crop_size'],
                                                         config['train_unsupervised']['crop_size'],
                                                         config['train_supervised']['base_size'],
                                                         config['train_unsupervised']['base_size']))

        logger.info("Current batch: {} [world]".format(int(config['train_unsupervised']
                                                           ['batch_size']) * args.world_size +
                                                       int(config['train_supervised']
                                                           ['batch_size']) * args.world_size))

        logger.info(
            "Current unsupervised loss function: {}, with weight {} and length {}".format(config['model']['un_loss'],
                                                                                          config['unsupervised_w'],
                                                                                          config['ramp_up']))

        logger.info("Use pre-trained model or no: {}".format("Yes" if config['state_dir'] is not None else "No"))

        logger.info("Current config+args: \n{}".format({**config, **vars(args)}))
    if args.ddp:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=args.local_rank,
            world_size=args.world_size
        )

    random.seed(28)
    torch.manual_seed(28)
    torch.cuda.manual_seed_all(28)
    torch.backends.cudnn.benchmark = True

    # DATA LOADERS
    config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    config['train_unsupervised']['weak_times'] = config['weak_times']

    config['train_supervised']['num_classes'] = config['num_classes']
    config['train_unsupervised']['num_classes'] = config['num_classes']
    config['train_supervised']['reflect_index'] = config['reflect_index']
    config['train_unsupervised']['reflect_index'] = config['reflect_index']
    config['val_loader']['reflect_index'] = config['reflect_index']
    config['val_loader']['num_classes'] = config['num_classes']

    supervised_loader = Pathology(config['train_supervised'], ddp_training=args.ddp)
    unsupervised_loader = Pathology(config['train_unsupervised'], ddp_training=args.ddp)

    val_loader = Pathology(config['val_loader'])

    iter_per_epoch = len(unsupervised_loader)

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'dice':
        # sup_loss = dice_loss
        sup_loss = dice_ce_loss
    else:
        raise NotImplementedError

    # UNSUPERVISED LOSS
    if config['model']['un_loss'] == 'semi_dice':
        unsup_loss = semi_dice_loss
    else:
        raise NotImplementedError

    # FEATURE LOSS
    if config['model']['f_loss'] == 'mse':
        f_loss = mse_loss
    else:
        raise NotImplementedError

    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                      rampup_starts=0, rampup_ends=config['ramp_up'],
                                      ramp_type="cosine_rampup")

    cons_w_f = consistency_weight(final_w=config['feature_w'], iters_per_epoch=len(unsupervised_loader),
                                  rampup_starts=0, rampup_ends=config['ramp_up'],
                                  ramp_type="cosine_rampup")

    w_vat = consistency_weight(final_w=config['vat_w'], iters_per_epoch=len(unsupervised_loader),
                               rampup_starts=0, rampup_ends=config['ramp_up'],
                               ramp_type="cosine_rampup")

    if args.architecture == "unet":
        Model = model_deep
        config['model']['data_d_h_w'] = [config['train_supervised']['crop_size'],
                                         config['train_supervised']['crop_size'],
                                         config['train_supervised']['crop_size']]
    else:
        raise NotImplementedError

    model = Model(in_channel=config['in_channel'], num_classes=config['num_classes'], state_dir=config['state_dir'],
                  sup_loss=sup_loss, cons_w_unsup=cons_w_unsup, unsup_loss=unsup_loss, f_loss=f_loss, cons_w_f=cons_w_f,
                  w_vat=w_vat)

    if args.local_rank <= 0:
        wandb_run = Tensorboard(config=config, online=True)

    trainer = Trainer(model=model,
                      config=config,
                      supervised_loader=supervised_loader,
                      unsupervised_loader=unsupervised_loader,
                      val_loader=val_loader,
                      iter_per_epoch=iter_per_epoch,
                      train_logger=logger if args.local_rank <= 0 else None,
                      wandb_run=wandb_run if args.local_rank <= 0 else None,
                      args=args)

    trainer.train()
    if args.local_rank <= 0:
        wandb_run.finish()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')

    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--epochs', default=-1, type=int)

    parser.add_argument('--warm_up', default=0, type=int)

    parser.add_argument('--labeled_examples', default=30, type=int)

    parser.add_argument('--weak_times', default=2, type=int)

    # parser.add_argument('-lr', '--learning-rate', default=4.5e-3, type=float,
    #                     help='Default HEAD Learning same as Others'
    #                          '*Note: in ddp training, lr will automatically times by n_gpu')

    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')

    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("-a", "--architecture", default='unet', type=str,
                        help="pick a architecture, only [unet] now")

    parser.add_argument("--ddp", action="store_true",
                        help="distributed data parallel training or not;"
                             "MUST SPECIFIED")

    parser.add_argument('--semi_p_th', type=float, default=0.6,
                        help='positive_threshold for semi-supervised loss')

    parser.add_argument('--semi_n_th', type=float, default=0.6,
                        help='negative_threshold for semi-supervised loss')

    parser.add_argument("-s", "--state_dir", default=None, type=str,
                        help="use pre-trained model to initialise entire model")

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    if args.architecture == "unet":
        config = json.load(open("./configs/config_unet.json"))
    else:
        raise NotImplementedError

    config['n_gpu'] = args.gpus

    if args.epochs == -1:
        args.epochs = config['trainer']['epochs']
    else:
        config['trainer']['epochs'] = args.epochs

    config['ramp_up'] = math.ceil(config['trainer']['epochs'] * 1 / 4)

    config['train_supervised']['batch_size'] = args.batch_size
    config['train_unsupervised']['batch_size'] = args.batch_size
    config['model']['warm_up_epoch'] = args.warm_up
    config['n_labeled_examples'] = args.labeled_examples
    config['weak_times'] = args.weak_times
    config['state_dir'] = args.state_dir

    args.ddp = True if args.gpus > 1 else False

    # we fix learning rate here in config
    # config['optimizer']['args']['lr'] = args.learning_rate

    if args.ddp:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9901'

    if args.ddp:
        mp.spawn(main, nprocs=config['n_gpu'], args=(config['n_gpu'], config, args))
    else:
        main(-1, 1, config, args)
