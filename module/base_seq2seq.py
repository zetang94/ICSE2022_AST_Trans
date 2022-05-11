import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset import make_std_mask
from module import process_data
from utils import UNK, BOS, PAD

__all__ = ['BaseTrans', 'GreedyGenerator']


class BaseTrans(nn.Module):
    def __init__(self):
        super(BaseTrans, self).__init__()

    def base_process(self, data):
        process_data(data)

        src_seq = data.src_seq
        data.src_mask = src_seq.eq(PAD)
        data.src_emb = self.src_embedding(src_seq)

        if data.tgt_seq is not None:
            tgt_seq = data.tgt_seq
            data.tgt_mask = make_std_mask(tgt_seq, PAD)
            data.tgt_emb = self.tgt_embedding(tgt_seq)

    def process_data(self, data):
        self.base_process(data)

    def forward(self, data):
        self.process_data(data)

        encoder_outputs = self.encode(data)
        decoder_outputs, attn_weights = self.decode(data, encoder_outputs)
        out = self.generator(decoder_outputs)
        return out

    def encode(self, data):
        return self.encoder(data)

    def decode(self, data, encoder_outputs):
        tgt_emb = data.tgt_emb
        tgt_mask = data.tgt_mask
        src_mask = data.src_mask

        tgt_emb = tgt_emb.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        tgt_mask = tgt_mask.repeat(self.num_heads, 1, 1)
        outputs, attn_weights = self.decoder(tgt=tgt_emb, memory=encoder_outputs,
                                             tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
        outputs = outputs.permute(1, 0, 2)
        return outputs, attn_weights


class GreedyGenerator(nn.Module):
    def __init__(self, model, max_tgt_len, multi_gpu=False):
        super(GreedyGenerator, self).__init__()
        if multi_gpu:
            self.model = model.module
        else:
            self.model = model
        self.max_tgt_len = max_tgt_len
        self.start_pos = BOS
        self.unk_pos = UNK

    def forward(self, data):
        data.tgt_seq = None
        self.model.process_data(data)

        encoder_outputs = self.model.encode(data)

        batch_size = encoder_outputs.size(0)
        ys = torch.ones(batch_size, 1).fill_(self.start_pos).long().to(encoder_outputs.device)
        for i in range(self.max_tgt_len - 1):
            data.tgt_mask = make_std_mask(ys, 0)
            data.tgt_emb = self.model.tgt_embedding(Variable(ys))
            decoder_outputs, decoder_attn = self.model.decode(data, encoder_outputs)
            out = self.model.generator(decoder_outputs)
            out = out[:, -1, :]
            _, next_word = torch.max(out, dim=1)
            ys = torch.cat([ys,
                            next_word.unsqueeze(1).long().to(encoder_outputs.device)], dim=1)

        return ys[:, 1:]
