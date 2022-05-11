import math
import torch
import torch.nn as nn
from module import _get_clones, transpose_for_scores

__all__ = ['FastMultiHeadedAttention']


class FastMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(FastMultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linear_layers = _get_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, start_nodes=None, end_nodes=None,
                rel_q=None, rel_k=None, rel_v=None):
        """relative q shape [1, 2k+1, dim]"""
        """relative v shape [1, 2k+1, dim]"""
        """"""
        query, key, value = [transpose_for_scores(l(x), self.h) for l, x in zip(self.linear_layers,
                                                                                (query, key, value))]

        output = self.rel_attn(query, key, value,
                               start_nodes=start_nodes, end_nodes=end_nodes,
                               rel_q=rel_q, rel_k=rel_k, rel_v=rel_v)

        output = output.permute(0, 2, 1, 3).contiguous()
        new_value_shape = output.size()[:-2] + (-1,)
        output = output.view(*new_value_shape)
        output = self.linear_layers[-1](output)

        return output, None

    @staticmethod
    def rel_attn(q, k, v, start_nodes, end_nodes, rel_q, rel_k, rel_v):
        """
        :param q: [batch_size, num_heads, seq_len, d_k]
        :param k: [batch_size, num_heads, seq_len, d_k]
        :param v: [batch_size, num_heads, seq_len, d_k]
        :param start_nodes: [batch_size, num_heads, k+1, seq_len]
        :param end_nodes: [batch_size, num_heads, k+1, seq_len]
        :param rel_q: [1, num_heads, 2k+2, d_k]
        :param rel_k: [1, num_heads, 2k+2, d_k]
        :param rel_v: [1, num_heads, 2k+2, d_k]
        :return:
        """
        batch_size, num_heads, seq_len, d_k = q.size()
        max_rel_pos = start_nodes.size(2)
        scale_factor = 1

        q_weights = q.contiguous().view(-1, d_k)
        k_weights = k.contiguous().view(-1, d_k)
        v_weights = v.contiguous().view(-1, d_k)

        start_node_mask = start_nodes.eq(-1)

        start_nodes += start_node_mask
        mask_matrix = torch.cat([start_node_mask, start_node_mask], dim=-2)
        mask_matrix = mask_matrix.view(batch_size, num_heads, -1)

        mask_matrix[:, :, :seq_len] = True

        map_pos = torch.arange(batch_size * num_heads, dtype=torch.long, device=start_nodes.device)
        map_pos = map_pos.unsqueeze(-1) * seq_len

        query_indexes = torch.cat([end_nodes, start_nodes], dim=-2)
        key_indexes = torch.cat([start_nodes, end_nodes], dim=-2)

        query_indexes = query_indexes.view(batch_size * num_heads, -1)
        key_indexes = key_indexes.view(batch_size * num_heads, -1)

        query_indexes += map_pos
        key_indexes += map_pos

        # query and key vec of context. shape [batch_size * num_heads * (k+1) * 2, seq_len, d_k]
        q_context = torch.embedding(q_weights, query_indexes)
        k_context = torch.embedding(k_weights, key_indexes)
        v_context = torch.embedding(v_weights, key_indexes)

        q_context = q_context.view(batch_size, num_heads, -1, seq_len, d_k)
        k_context = k_context.view(batch_size, num_heads, -1, seq_len, d_k)
        v_context = v_context.view(batch_size, num_heads, -1, seq_len, d_k)

        # context -> context
        c2c = torch.mul(q_context, k_context).sum(dim=-1)
        scores = c2c

        # context -> position
        if rel_k is not None:
            k_pos = rel_k.unsqueeze(-2)
            c2p = torch.mul(q_context, k_pos).sum(dim=-1)
            scores += c2p
            scale_factor += 1

        # position -> context
        if rel_q is not None:
            q_pos = rel_q.unsqueeze(-2)
            p2c = torch.mul(q_pos, k_context).sum(dim=-1)
            scores += p2c
            scale_factor += 1

        # scores = (c2c + c2p + p2c) / (3 * sqrt(d_k))
        scores = scores / (scale_factor * math.sqrt(d_k))
        scores = scores.view(batch_size, num_heads, -1)

        # score shape [batch_size, num_heads, 2(k+1) * seq_len]
        # index shape [batch_size, num_heads, 2(k+1) * seq_len]
        query_indexes = query_indexes - map_pos
        query_indexes = query_indexes.view(batch_size, num_heads, -1)

        scores = scores.masked_fill(mask_matrix, -1e9)

        scores = torch.exp(scores)

        score_sum = torch.zeros(scores.size(), device=scores.device)
        score_sum.scatter_add_(dim=-1, index=query_indexes, src=scores)

        score_sum += score_sum.eq(0) * -1e9
        score_sum = torch.gather(score_sum, index=query_indexes, dim=-1)

        # [batch_size, num_heads, 2(k+1), seq_len, 1]
        scores = (scores / score_sum).view(batch_size, num_heads, -1, seq_len, 1)

        # shape [batch_size, num_heads, 2(k+1), seq_len, d_k]
        if rel_v is not None:
            v_context += rel_v.unsqueeze(-2)
        attn_v = torch.mul(scores, v_context)

        output = torch.zeros(attn_v.size(), device=attn_v.device)
        query_indexes = query_indexes.view(batch_size, num_heads, -1, seq_len).unsqueeze(-1)
        query_indexes = query_indexes.repeat_interleave(repeats=d_k, dim=-1)
        output.scatter_add_(dim=-2, index=query_indexes, src=attn_v)

        output = output.sum(dim=-3)

        return output






