from Utils.losses import *
from itertools import chain
from Base.base_model import BaseModel
from Model.encoder_decoder import *


class EntireModel(BaseModel):
    def __init__(self, in_channel, num_classes, sup_loss=None, cons_w_unsup=None, unsup_loss=None):
        super(EntireModel, self).__init__()
        self.encoder_t = EncoderNetwork(in_channel)
        self.decoder_t = TDecoderNetwork(num_classes)
        self.encoder_s = EncoderNetwork(in_channel)
        self.decoder_s = VATDecoderNetwork(num_classes)
        self.mode = "semi"
        self.sup_loss = sup_loss
        self.unsup_loss_w = cons_w_unsup
        self.unsuper_loss = unsup_loss
        self.num_classes = num_classes

    def freeze_teachers_parameters(self):
        for p in self.encoder_t.parameters():
            p.requires_grad = False
        for p in self.decoder_t.parameters():
            p.requires_grad = False

    def warm_up_forward(self, id, x, y):
        if id == 1:
            f1, f2, f3, f4, f5 = self.encoder_t(x)
            output_l = self.decoder_t(f1, f2, f3, f4, f5)
        else:
            f1, f2, f3, f4, f5 = self.encoder_s(x)
            output_l = self.decoder_s(f1, f2, f3, f4, f5)

        # unique, count = np.unique(y.cpu().numpy(), return_counts=True)
        # data_count = dict(zip(unique, count))
        # print(data_count)

        loss = self.sup_loss(output_l, y, num_classes=self.num_classes)
        curr_losses = {'loss_sup': loss}
        outputs = {'sup_pred': output_l}
        return loss, curr_losses, outputs

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, id=0,
                warm_up=False, semi_p_th=0.6, semi_n_th=0.6):
        if warm_up:
            return self.warm_up_forward(id=id, x=x_l, y=target_l)
        f1, f2, f3, f4, f5 = self.encoder_s(x_l)
        output_l = self.decoder_s(f1, f2, f3, f4, f5, t_model=self.decoder_t)
        # output_l = self.decoder_s(self.encoder_s(x_l), t_model=self.decoder_t)
        # Supervised loss
        loss_sup = self.sup_loss(output_l, target_l, num_classes=self.num_classes)

        # print(loss_sup)

        curr_losses = {'loss_sup': loss_sup}
        # output_ul = self.decoder_s(self.encoder_s(x_ul), t_model=self.decoder_t)
        f1, f2, f3, f4, f5 = self.encoder_s(x_ul)
        output_ul = self.decoder_s(f1, f2, f3, f4, f5, t_model=self.decoder_t)
        loss_unsup, pass_rate, neg_loss = self.unsuper_loss(inputs=output_ul, targets=target_ul,
                                                            conf_mask=True, threshold=semi_p_th,
                                                            threshold_neg=semi_n_th, num_classes=self.num_classes)

        # for negative learning
        if semi_n_th > .0:
            confident_reg = .5 * torch.mean(F.softmax(output_ul, dim=1) ** 2)
            loss_unsup += neg_loss
            loss_unsup += confident_reg

        loss_unsup = loss_unsup * self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
        total_loss = loss_unsup + loss_sup

        curr_losses['loss_unsup'] = loss_unsup
        curr_losses['pass_rate'] = pass_rate
        curr_losses['neg_loss'] = neg_loss
        outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}
        return total_loss, curr_losses, outputs
