import torch
import torch.nn as nn
from module import Embeddings, _get_clones, FastRelEmbeddings, FeedForward, SublayerConnection
from module.base_seq2seq import BaseTrans
from module.components import DecoderLayer, BaseDecoder, Generator
from module import FastMultiHeadedAttention


class FastASTTrans(BaseTrans):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size, par_heads, num_heads,
                 max_rel_pos, pos_type, num_layers, dim_feed_forward, dropout, state_dict=None):
        super(FastASTTrans, self).__init__()
        self.num_heads = num_heads 
        bro_heads = num_heads - par_heads
        self.pos_type = pos_type.split('_')

        self.src_embedding = Embeddings(hidden_size=hidden_size,
                                        vocab_size=src_vocab_size,
                                        dropout=dropout,
                                        with_pos=False)

        self.tgt_embedding = Embeddings(hidden_size=hidden_size,
                                        vocab_size=tgt_vocab_size,
                                        dropout=dropout,
                                        with_pos=True)

        encoder_layer = FastASTEncoderLayer(hidden_size, self.num_heads, dim_feed_forward, dropout)
        self.encoder = FastASTEncoder(encoder_layer, num_layers, par_heads, bro_heads, self.pos_type,
                                      max_rel_pos, hidden_size, dropout=dropout)

        decoder_layer = DecoderLayer(hidden_size, self.num_heads, dim_feed_forward, dropout, activation="gelu")
        self.decoder = BaseDecoder(decoder_layer, num_layers, norm=nn.LayerNorm(hidden_size))
        self.generator = Generator(tgt_vocab_size, hidden_size, dropout)

        print('Init or load model.')
        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.load_state_dict(state_dict)


class FastASTEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, par_heads, bro_heads, pos_type, max_rel_pos,
                 hidden_size, dropout=0.2):
        super(FastASTEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.par_heads = par_heads
        self.bro_heads = bro_heads
        d_k = hidden_size // (par_heads + bro_heads)
        if par_heads > 0:
            self.par_rel_emb = FastRelEmbeddings(d_k, par_heads, max_rel_pos, pos_type, dropout=dropout)
        if bro_heads > 0:
            self.bro_rel_emb = FastRelEmbeddings(d_k, bro_heads, max_rel_pos, pos_type, dropout=dropout)

        self.end_nodes = None

    def forward(self, data):
        output = data.src_emb
        rel_par_pos = data.par_edges
        rel_bro_pos = data.bro_edges

        batch_size, max_rel_pos, max_ast_len = rel_par_pos.size()
        rel_par_q, rel_par_k, rel_par_v = None, None, None
        rel_bro_q, rel_bro_k, rel_bro_v = None, None, None
        if self.par_heads > 0:
            rel_par_q, rel_par_k, rel_par_v = self.par_rel_emb()
        if self.bro_heads > 0:
            rel_bro_q, rel_bro_k, rel_bro_v = self.bro_rel_emb()
        rel_q = self.concat_vec(rel_par_q, rel_bro_q, dim=1)
        rel_k = self.concat_vec(rel_par_k, rel_bro_k, dim=1)
        rel_v = self.concat_vec(rel_par_v, rel_bro_v, dim=1)

        start_nodes = self.concat_pos(rel_par_pos, rel_bro_pos)

        need_end_nodes = True
        if self.end_nodes is not None and batch_size == self.end_nodes.size(0):
            need_end_nodes = False

        if need_end_nodes:
            end_nodes = torch.arange(max_ast_len, device=start_nodes.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.end_nodes = end_nodes.repeat(batch_size, self.par_heads + self.bro_heads,
                                              max_rel_pos, 1)

        for i, layer in enumerate(self.layers):
            output = layer(output, start_nodes, self.end_nodes, rel_q, rel_k, rel_v)

        return self.norm(output)

    def concat_pos(self, rel_par_pos, rel_bro_pos):
        if self.par_heads == 0:
            return rel_bro_pos.unsqueeze(1).repeat_interleave(repeats=self.bro_heads,
                                                              dim=1)
        if self.bro_heads == 0:
            return rel_par_pos.unsqueeze(1).repeat_interleave(repeats=self.par_heads,
                                                              dim=1)

        rel_par_pos = rel_par_pos.unsqueeze(1).repeat_interleave(repeats=self.par_heads,
                                                                 dim=1)
        rel_bro_pos = rel_bro_pos.unsqueeze(1).repeat_interleave(repeats=self.bro_heads,
                                                                 dim=1)
        rel_pos = self.concat_vec(rel_par_pos, rel_bro_pos, dim=1)

        return rel_pos

    @staticmethod
    def concat_vec(vec1, vec2, dim):
        if vec1 is None:
            return vec2
        if vec2 is None:
            return vec1
        return torch.cat([vec1, vec2], dim=dim)


class FastASTEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dim_feed_forward, dropout):
        super(FastASTEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.self_attn = FastMultiHeadedAttention(num_heads, hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, dim_feed_forward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.sublayer = _get_clones(SublayerConnection(hidden_size, dropout), 2)

    def forward(self, src, start_nodes, end_nodes, rel_q, rel_k, rel_v):
        src, attn_weights = self.sublayer[0](src, lambda x: self.self_attn(x, x, x, start_nodes, end_nodes,
                                                                           rel_q, rel_k, rel_v))
        src, _ = self.sublayer[1](src, self.feed_forward)
        return src
