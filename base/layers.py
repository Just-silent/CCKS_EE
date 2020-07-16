# -*- coding: utf-8 -*-
# @Author  : star
# @Time    : 2019/9/10 10:09
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    """
    双向LSTM(用于embedding层是BERT的)
    """

    def __init__(self, embedding_size=768, hidden_dim=512, layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_size, hidden_dim // 2, num_layers=layers, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size=None):
        '''
        初始化隐层
        :param batch_size:
        :return:
        '''
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        return h0, c0

    def forward(self, embeds, sent_lens):
        """
        定义前向传播
        :param embeds: embedding层的输出
        :param sent_lens: 批次句子长度集合
        :return:
        """
        embeds = pack_padded_sequence(embeds, sent_lens)  # 移除填充
        self.hidden = self.init_hidden(batch_size=len(sent_lens))
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)  # 填充 一个tuple，包含被填充后的序列，和batch中 序列长度的 列表。
        assert torch.equal(sent_lens, new_batch_size)
        return lstm_out


class Linears(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hiddens,
                 bias=True,
                 activation='tanh'):
        super(Linears, self).__init__()
        assert len(hiddens) > 0

        self.in_features = in_features
        self.out_features = self.output_size = out_features

        in_dims = [in_features] + hiddens[:-1]
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=bias)
                                      for in_dim, out_dim
                                      in zip(in_dims, hiddens)])
        self.output_linear = nn.Linear(hiddens[-1], out_features, bias=bias)
        self.activation = getattr(torch.functional, activation)

    def forward(self, inputs):
        linear_outputs = inputs
        for linear in self.linears:
            linear_outputs = linear.forward(linear_outputs)
            linear_outputs = self.activation(linear_outputs)
        return self.output_linear.forward(linear_outputs)

#Reused from https://blog.csdn.net/uhauha2929/article/details/81951760
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        # outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1))
        return outputs, weights


# Reused from https://github.com/JayParks/transformer/
class ScaledDotProductAttention(nn.Module):
    def  __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # torch.bmm(batch1,batch2) 两个批(batch1,batch2)内的矩阵,进行 "批矩阵乘" 操作
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            print(attn_mask.size(), attn.size())
            assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask, -float('inf'))

        # 计算softmax
        attn = self.softmax(attn)
        attn = self.dropout(attn) # 添加dropout
        outputs = torch.bmm(attn, v)  # 和V做点积 outputs: [b_size x len_q x d_v] 乘以值向量，目的是弱化相关词，得出加权值向量
        return outputs, attn


# 层归一化
class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean.expand_as(z)) / (std.expand_as(z) + self.eps)
        ln_out = self.gamma.expand_as(ln_out) * ln_out + self.beta.expand_as(ln_out)
        return ln_out


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k), requires_grad=True)
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k), requires_grad=True)
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_v), requires_grad=True)

        self.attention = ScaledDotProductAttention(d_k, dropout)

        # 权重初始化，服从正态分布的权重初始化
        init.xavier_normal_(self.w_q)
        init.xavier_normal_(self.w_k)
        init.xavier_normal_(self.w_v)

    def forward(self, q, k, v, attn_mask=None):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        # repeat 沿着指定的维度重复tensor，重复n_heads次
        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)  # [b_size * n_heads x len_v x d_v]

        # perform attention, result_size = [b_size * n_heads x len_q x d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)
        # 以上步骤得出multi-head的 q,k,v向量
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        # return a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):  # (d_k,d_v,d_model)=(lstm_out,lstm_out,lstm_out)
        super(MultiHeadAttention, self).__init__()
        self.attention = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.proj = nn.Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)  # 层归一化

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask)
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1)  # 连接
        # project back to residual size, result_size = [b_size x len_q x d_model]
        outputs = self.proj(outputs)  # 映射
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs), attn

