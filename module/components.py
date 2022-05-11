import copy
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn

__all__ = ['_get_clones', 'FeedForward', 'Embeddings',
           'build_relative_position', 'transpose_for_scores', 'SublayerConnection',
           'c2p_dynamic_expand', 'p2c_dynamic_expand', 'pos_dynamic_expand', 'PositionalEncoding',
           'FastRelEmbeddings', 'DecoderLayer', 'BaseDecoder', 'Generator', 'process_data']


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


class FeedForward3(nn.Module):
    def __init__(self, d_model, dim_feed_forward, dropout=0.1):
        super(FeedForward3, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feed_forward)
        self.linear2 = nn.Linear(dim_feed_forward, int(d_model/2))
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feed_forward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feed_forward)
        self.linear2 = nn.Linear(dim_feed_forward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x)))), None


class Embeddings(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout=0.1, with_pos=False):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if with_pos:
            self.pos_emb = PositionalEncoding(hidden_size)
        else:
            self.pos_emb = None
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        words_embeddings = self.word_embeddings(x)
        if self.pos_emb is not None:
            words_embeddings = self.pos_emb(words_embeddings)

        embeddings = self.norm(words_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) *
                             -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        output, attn_weights = sublayer(self.norm(x))
        return x + self.dropout(output), attn_weights


class RelEmbeddings(nn.Module):
    def __init__(self, d_model, num_heads, k, pos_type, dropout=0.0):
        super(RelEmbeddings, self).__init__()

        self.d_model = d_model
        self.k = 2 * k + 2
        self.pos_type = pos_type
        self.num_heads = num_heads
        if 'p2q' in pos_type:
            self.rel_emb_q = nn.Embedding(self.k, d_model, padding_idx=self.k // 2)  # pad id=k+1 -> zero
        if 'p2k' in pos_type:
            self.rel_emb_k = nn.Embedding(self.k, d_model, padding_idx=self.k // 2)
        if 'p2v' in pos_type:
            self.rel_emb_v = nn.Embedding(self.k, d_model, padding_idx=self.k // 2)
        self.dropout = nn.Dropout(dropout)

    def get_rel_weights(self, rel_params):
        rel_params = rel_params * math.sqrt(self.d_model)
        rel_params = self.dropout(rel_params)

        rel_params = rel_params.unsqueeze(0).unsqueeze(0)
        rel_params = rel_params.repeat(1, self.num_heads, 1, 1)

        return rel_params

    def get_p2v_emb(self, inputs):
        if 'p2v' in self.pos_type:
            rel_v = self.rel_emb_v(inputs) * math.sqrt(self.d_model)
            rel_v = self.dropout(rel_v)
            rel_v = rel_v.repeat(1, 1, 1, self.num_heads)
            return rel_v
        else:
            return None


class DebertaRelEmbeddings(RelEmbeddings):
    def __init__(self, d_model, num_heads, k, pos_type, dropout=0.0):
        super(DebertaRelEmbeddings, self).__init__(d_model, num_heads, k, pos_type, dropout)

    def forward(self, inputs):
        rel_q, rel_k, rel_v = None, None, None
        if 'p2q' in self.pos_type:
            rel_q = self.get_rel_weights(self.rel_emb_q.weight)
        if 'p2k' in self.pos_type:
            rel_k = self.get_rel_weights(self.rel_emb_k.weight)
        if 'p2v' in self.pos_type:
            rel_v = self.get_p2v_emb(inputs)

        return rel_q, rel_k, rel_v


class FastRelEmbeddings(RelEmbeddings):
    def __init__(self, d_model, num_heads, k, pos_type, dropout=0.0):
        super(FastRelEmbeddings, self).__init__(d_model, num_heads, k, pos_type, dropout)

    def forward(self):
        rel_q, rel_k, rel_v = None, None, None
        if 'p2q' in self.pos_type:
            rel_q = self.get_rel_weights(self.rel_emb_q.weight)
        if 'p2k' in self.pos_type:
            rel_k = self.get_rel_weights(self.rel_emb_k.weight)
        if 'p2v' in self.pos_type:
            rel_v = self.get_rel_weights(self.rel_emb_v.weight)

        return rel_q, rel_k, rel_v


def build_relative_position(query_size, key_size, max_relative_positions, device, need_traverse=False):
    """
    :return: obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """
    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    if need_traverse:
        rel_pos_ids = -rel_pos_ids
    rel_pos_ids += max_relative_positions + 1
    rel_pos_ids = torch.clamp(rel_pos_ids, 1, 2 * max_relative_positions + 1)
    return rel_pos_ids


def transpose_for_scores(x, num_heads):
    new_x_shape = x.size()[:-1] + (num_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)


class Generator(nn.Module):
    def __init__(self, tgt_vocab_size, hidden_size, dropout):
        super(Generator, self).__init__()
        self.soft_max = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, outputs):
        out = self.linear(outputs)
        gen_prob = self.soft_max(self.dropout(out))
        return torch.log(gen_prob)


def process_data(data):
    batch_size = data.num_graphs
    for key in data.keys:
        new_value_shape = (batch_size, -1) + data[key].size()[1:]
        data[key] = data[key].view(*new_value_shape)


class BaseDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(BaseDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output, attn_weights = mod(output, memory, tgt_mask=tgt_mask,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout=dropout)
        self.sublayer = _get_clones(SublayerConnection(d_model, dropout), 3)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(DecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt, attn_weights = self.sublayer[0](tgt, lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask,
                                                                           key_padding_mask=tgt_key_padding_mask))

        tgt, attn_weights = self.sublayer[1](tgt, lambda x: self.multihead_attn(
            x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        ))

        tgt, _ = self.sublayer[2](tgt, self.feed_forward)
        return tgt, attn_weights


