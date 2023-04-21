import torch.nn as nn

from Model.encoder_decoder import EncoderNetwork, TDecoderNetwork


class TeacherModel(nn.Module):
    def __init__(self, encoder: EncoderNetwork, decoder: TDecoderNetwork):
        super(TeacherModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        f1, f2, f3, f4, f5 = self.encoder(data)
        pred = self.decoder(f1, f2, f3, f4, f5)
        return pred
