import torch
from tqdm import tqdm
from Utils.ramps import *
from itertools import cycle
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from Base.base_trainer import BaseTrainer
from Utils.sliding_evaluator import SlidingEval
from Utils.metrics import AverageMeter
from medpy.metric.binary import dc, hd95, sensitivity
from Utils.param_trans import trans_predict, trans_type
from Utils.losses import dice_loss


class Trainer(BaseTrainer):
    def __init__(self, model, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None, wandb_run=None, args=None):
        super(Trainer, self).__init__(model, config, iter_per_epoch, train_logger, args)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.args = args
        self.tensor_board = wandb_run if self.args.local_rank <= 0 else None
        self.iter_per_epoch = iter_per_epoch
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1
        self.num_classes = config['num_classes']
        self.weak_times = config['weak_times']
        self.mode = self.model.module.mode
        self.evaluator = SlidingEval(model=self.model,
                                     crop_size=config['train_unsupervised']['crop_size'],
                                     stride_rate=2 / 3,
                                     device="cuda:0" if self.args.local_rank < 0 else
                                     "cuda:{}".format(self.args.local_rank),
                                     class_number=self.num_classes)

    @torch.no_grad()
    def update_teachers(self, teacher_encoder, teacher_decoder, keep_rate=0.99):
        student_encoder_dict = self.model.module.encoder_s.state_dict()
        student_decoder_dict = self.model.module.decoder_s.state_dict()
        new_teacher_encoder_dict = OrderedDict()
        new_teacher_decoder_dict = OrderedDict()

        for key, value in teacher_encoder.state_dict().items():

            if key in student_encoder_dict.keys():
                new_teacher_encoder_dict[key] = (
                        student_encoder_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student encoder model".format(key))

        for key, value in teacher_decoder.state_dict().items():

            if key in student_decoder_dict.keys():
                new_teacher_decoder_dict[key] = (
                        student_decoder_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student decoder model".format(key))
        teacher_encoder.load_state_dict(new_teacher_encoder_dict, strict=True)
        teacher_decoder.load_state_dict(new_teacher_decoder_dict, strict=True)

    def predict_with_out_grad(self, images):
        with torch.no_grad():
            predict_target_ul = torch.zeros(images.shape[0], self.num_classes, images.shape[2], images.shape[3],
                                            images.shape[4]).cuda(non_blocking=True)
            features = []
            for i in range(self.weak_times):
                f1, f2, f3, f4, f5 = self.model.module.encoder_t(images[:, i, ...].unsqueeze(dim=1))
                if len(features) == 0:
                    features.append(f1.clone())
                    features.append(f2.clone())
                    features.append(f3.clone())
                    features.append(f4.clone())
                    features.append(f5.clone())
                else:
                    features[0] += f1.clone()
                    features[1] += f2.clone()
                    features[2] += f3.clone()
                    features[3] += f4.clone()
                    features[4] += f5.clone()
                predict_target_ul = predict_target_ul + self.model.module.decoder_t(f1, f2, f3, f4, f5)

            for feature in features:
                feature /= self.weak_times
            predict_target_ul /= self.weak_times
            if predict_target_ul.shape[-3:] != images.shape[-3:]:
                # predict_target_ul = torch.nn.functional.interpolate(predict_target_ul,
                #                                                     size=(
                #                                                         image.shape[-3], image.shape[-2],
                #                                                         image.shape[-1]),
                #                                                     mode='bilinear',
                #                                                     align_corners=True)
                raise ValueError

        return predict_target_ul, features

    def _warm_up(self, epoch, id):
        self.model.train()
        assert id == 1 or id == 2, "Expect ID in 1 or 2"
        dataloader = iter(self.supervised_loader)
        tbar = range(len(self.supervised_loader))

        if self.args.ddp:
            self.supervised_loader.sampler.set_epoch(epoch=epoch - 1)

        tbar = tqdm(tbar, ncols=135) if self.args.local_rank <= 0 else tbar
        self._reset_metrics()
        for batch_idx in tbar:
            (input_l_wk, input_l_str, target_l) = next(dataloader)

            input_l = input_l_wk if id == 1 else input_l_str

            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)

            total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=None,
                                                         target_ul=None,
                                                         curr_iter=batch_idx, epoch=epoch - 1, id=id, warm_up=True)
            if id == 1:
                self.optimizer_t.zero_grad()
            else:
                self.optimizer_s.zero_grad()

            total_loss = total_loss.mean()
            total_loss.backward()
            if id == 1:
                self.optimizer_t.step()
            else:
                self.optimizer_s.step()

            self._update_losses(cur_losses)
            self._compute_evaluation_index(outputs, target_l, None, sup=True)
            _ = self._log_values(cur_losses)

            del input_l, target_l
            del total_loss, cur_losses, outputs

            if self.args.local_rank <= 0:
                tbar.set_description('ID {} Warm ({}) | Ls {:.2f} |'.format(id, epoch + 1, self.loss_sup.average))

        return

    def _train_epoch(self, epoch, id):
        assert id == 1, "Expect ID 1"
        self.model.module.freeze_teachers_parameters()
        self.model.train()
        if self.args.ddp:
            self.supervised_loader.sampler.set_epoch(epoch=epoch - 1)
            self.unsupervised_loader.sampler.set_epoch(epoch=epoch - 1)
        dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
        tbar = range(len(self.unsupervised_loader))

        tbar = tqdm(tbar, ncols=135, leave=True) if self.args.local_rank <= 0 else tbar
        self._reset_metrics()
        for batch_idx in tbar:
            if self.args.local_rank <= 0:
                self.tensor_board.step_forward(len(self.unsupervised_loader) * (epoch - 1) + batch_idx)
            if self.mode == "semi":
                (_, input_l, target_l), (input_ul_wk, input_ul_str, target_ul) = next(dataloader)
                input_ul_wk, input_ul_str, target_ul = input_ul_wk.cuda(non_blocking=True), \
                    input_ul_str.cuda(non_blocking=True), \
                    target_ul.cuda(non_blocking=True)
            else:
                (_, input_l, target_l), _ = next(dataloader)
                input_ul_wk, input_ul_str, target_ul = None, None, None

            # strong aug for all the supervised images
            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)

            # predicted unlabeled data
            if self.mode == "semi":
                t_prob, t_feat = self.predict_with_out_grad(input_ul_wk)

                predict_target_ul, predict_features_ul = t_prob, t_feat
            else:
                predict_target_ul, predict_features_ul = None, []

            origin_predict = predict_target_ul.detach().clone()

            #  Real-time recording of image prediction information
            # if batch_idx == 0 or batch_idx == int(len(self.unsupervised_loader) / 2):
            #     if self.args.local_rank <= 0:
            #         self.tensor_board.update_wandb_pathology_image(images=input_ul_wk,
            #                                                   ground_truth=target_ul,
            #                                                   teacher_prediction=predict_target_ul)

            # input_l, target_l, input_ul_str, predict_target_ul = self.cut_mix(input_l, target_l,
            #                                                                   input_ul_str,
            #                                                                   predict_target_ul)

            total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l,
                                                         x_ul=input_ul_str,
                                                         target_ul=predict_target_ul,
                                                         teacher_features=predict_features_ul,
                                                         curr_iter=batch_idx, epoch=epoch - 1, id=id,
                                                         semi_p_th=self.args.semi_p_th,
                                                         semi_n_th=self.args.semi_n_th)

            total_loss = total_loss.mean()

            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()
            outputs['unsup_pred'] = origin_predict
            self._update_losses(cur_losses)
            self._compute_evaluation_index(outputs, target_l, target_ul,
                                           sup=True if self.model.module.mode == "supervised" else False)

            _ = self._log_values(cur_losses)

            if self.args.local_rank <= 0:
                if batch_idx == 0 or batch_idx == int(len(self.unsupervised_loader) / 2):
                    self.tensor_board.update_table(cur_losses['pass_rate']['entire_prob_boundary'],
                                                   axis_name={"x": "boundary", "y": "rate"},
                                                   title="pass_in_each_boundary")

                    self.tensor_board.update_table(cur_losses['pass_rate']['max_prob_boundary'],
                                                   axis_name={"x": "boundary", "y": "rate"},
                                                   title="max_prob_in_each_boundary")

                if batch_idx % self.log_step == 0:
                    # for i, opt_group in enumerate(self.optimizer_s.param_groups[:2]):
                    #     self.tensor_board.upload_single_info({f"learning_rate_{i}": opt_group['lr']})
                    self.tensor_board.upload_single_info({f"learning_rate": self.optimizer_s.param_groups[0]['lr']})
                    self.tensor_board.upload_single_info({"ramp_up": self.model.module.unsup_loss_w.current_rampup})

                tbar.set_description('ID {} T ({}) | Ls {:.3f} Lu {:.3f} Lf {:.3f} Ds {:.3f} Du {:.3f}|'.format(
                    id, epoch, self.loss_sup.average, self.loss_unsup.average, self.loss_f.average, self.dc_l,
                    self.dc_ul))

            if self.args.ddp:
                dist.barrier()

            del input_l, target_l, input_ul_wk, input_ul_str, target_ul
            del total_loss, cur_losses, outputs

            self.lr_scheduler_s.step(epoch=epoch - 1)

            with torch.no_grad():
                self.update_teachers(teacher_encoder=self.model.module.encoder_t,
                                     teacher_decoder=self.model.module.decoder_t)
                if self.args.ddp:
                    dist.barrier()

        return

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.wrt_mode = 'val'
        total_loss_val = AverageMeter()
        stride = int(np.ceil(len(self.val_loader.dataset) / self.args.gpus))
        current_rank = max(0, self.args.local_rank)
        e_record = min((current_rank + 1) * stride, len(self.val_loader.dataset))
        shred_list = list(range(current_rank * stride, e_record))
        # total_inter, total_union = torch.tensor(0), torch.tensor(0)
        # total_correct, total_label = torch.tensor(0), torch.tensor(0)
        total_dc, total_hd95, total_sst = torch.tensor(.0), torch.tensor(.0), torch.tensor(.0)
        tbar = tqdm(shred_list, ncols=130, leave=True) if self.args.local_rank <= 0 else shred_list
        with torch.no_grad():
            for batch_idx in tbar:
                data, target = self.val_loader.dataset[batch_idx]

                target, data = torch.tensor(target).unsqueeze(0).cuda(non_blocking=True), torch.tensor(data).unsqueeze(
                    0).cuda(non_blocking=True)

                output = self.evaluator(img=data)
                output = torch.tensor(output, dtype=torch.float).cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # LOSS
                # loss = F.cross_entropy(output, target)
                loss = dice_loss(output, target, num_classes=self.num_classes)
                total_loss_val.update(loss.item())
                # correct, labeled, inter, union = eval_metrics(output, target, self.num_classes)

                cur_dc = dc(trans_predict(output), trans_type(target))
                cur_hd95 = hd95(trans_predict(output), trans_type(target))
                cur_sst = sensitivity(trans_predict(output), trans_type(target))

                # print(cur_dc, cur_hd95, cur_sst)

                total_dc = total_dc + cur_dc
                total_hd95 = total_hd95 + cur_hd95
                total_sst = total_sst + cur_sst

                # total_inter, total_union = total_inter + inter, total_union + union
                # total_correct, total_label = total_correct + correct, total_label + labeled

            if self.args.gpus > 1:
                # total_inter = torch.tensor(total_inter, device=self.args.local_rank)
                # total_union = torch.tensor(total_union, device=self.args.local_rank)
                # total_correct = torch.tensor(total_correct, device=self.args.local_rank)
                # total_label = torch.tensor(total_label, device=self.args.local_rank)
                total_dc = torch.tensor(total_dc, device=self.args.local_rank)
                total_hd95 = torch.tensor(total_hd95, device=self.args.local_rank)
                total_sst = torch.tensor(total_sst, device=self.args.local_rank)
                # dist.all_reduce(total_inter, dist.ReduceOp.SUM)
                # dist.all_reduce(total_union, dist.ReduceOp.SUM)
                # dist.all_reduce(total_correct, dist.ReduceOp.SUM)
                # dist.all_reduce(total_label, dist.ReduceOp.SUM)
                dist.all_reduce(total_dc, dist.ReduceOp.SUM)
                dist.all_reduce(total_hd95, dist.ReduceOp.SUM)
                dist.all_reduce(total_sst, dist.ReduceOp.SUM)

            # PRINT INFO
            # pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            # IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            # mIoU = IoU.mean().item()
            # pixAcc = pixAcc.item()

            v_dc = total_dc.item() / len(tbar)
            v_hd95 = total_hd95.item() / len(tbar)
            v_sst = total_sst.item() / len(tbar)

            seg_metrics = {"Dice": np.round(v_dc, 4),
                           "HD95": np.round(v_hd95, 4),
                           "Sensitivity": np.round(v_sst, 4)}

        if self.args.local_rank <= 0:
            print('EVAL EPOCH ({}) | Dice: {:.4f}, HD95: {:.4f}, Sensitivity: {:.4f} |'.format(
                epoch,
                v_dc,
                v_hd95,
                v_sst))
            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }
            valid_dict = {}
            for k, v in list(seg_metrics.items()):
                valid_dict[f'valid_{k}'] = v
            self.tensor_board.upload_wandb_info(valid_dict)

            return log

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_f = AverageMeter()
        self.dc_l, self.dc_ul = .0, .0
        self.hd95_l, self.hd95_ul = .0, .0
        self.sst_l, self.sst_ul = .0, .0

    def _update_losses(self, cur_losses):
        if "loss_sup" in cur_losses.keys():
            self.loss_sup.update(cur_losses['loss_sup'].mean().item())
        if "loss_unsup" in cur_losses.keys():
            self.loss_unsup.update(cur_losses['loss_unsup'].mean().item())
        if "loss_f" in cur_losses.keys():
            self.loss_f.update(cur_losses['loss_f'].mean().item())

    def _compute_evaluation_index(self, outputs, target_l, target_ul, sup=False):
        self.dc_l = dc(trans_predict(outputs['sup_pred']), trans_type(target_l))
        self.hd95_l = hd95(trans_predict(outputs['sup_pred']), trans_type(target_l))
        self.sst_l = sensitivity(trans_predict(outputs['sup_pred']), trans_type(target_l))

        if sup:
            return

        if self.mode == 'semi':
            self.dc_ul = dc(trans_predict(outputs['unsup_pred']), trans_type(target_ul))
            self.hd95_ul = hd95(trans_predict(outputs['unsup_pred']), trans_type(target_ul))
            self.sst_ul = sensitivity(trans_predict(outputs['unsup_pred']), trans_type(target_ul))

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average

        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average

        logs['dc_l'] = self.dc_l
        logs['hd95_l'] = self.hd95_l
        logs['sst_l'] = self.sst_l

        if self.args.local_rank <= 0:
            self.tensor_board.upload_single_info({'loss_sup': self.loss_sup.average})
            self.tensor_board.upload_single_info({'dc_l': self.dc_l})
            self.tensor_board.upload_single_info({'hd95_l': self.hd95_l})
            self.tensor_board.upload_single_info({'sst_l': self.sst_l})

            if self.mode == 'semi':
                logs['dc_ul'] = self.dc_ul
                logs['hd95_ul'] = self.hd95_ul
                logs['sst_ul'] = self.sst_ul
                self.tensor_board.upload_single_info({'loss_unsup': self.loss_unsup.average})
                self.tensor_board.upload_single_info({'dc_ul': self.dc_ul})
                self.tensor_board.upload_single_info({'hd95_ul': self.hd95_ul})
                self.tensor_board.upload_single_info({'sst_ul': self.sst_ul})

        return logs
