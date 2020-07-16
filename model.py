# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/4/26 12:45

import math
import torch
import random
import numpy as np
import torch.nn as nn
from torchcrf import CRF
from loss import DiceLoss
from config import device
from transformers import *
from torch.nn import functional as F
from torchtext.vocab import Vectors
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base.layers import ScaledDotProductAttention, SelfAttention, MultiHeadAttention


class TransformerEncoderModel(nn.Module):
    def __init__(self, config, ntoken, ntag, vectors):
        super(TransformerEncoderModel, self).__init__()
        self.config = config
        self.src_mask = None
        self.vectors = vectors
        self.embedding_size = config.embedding_size
        self.embedding = nn.Embedding(ntoken, config.embedding_size)
        self.pos_encoder = PositionalEncoding(config.embedding_size, config.dropout)
        encoder_layers = TransformerEncoderLayer(config.embedding_size, config.nhead, config.nhid, config.dropout)
        self.lstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.bi_lstm_hidden // 2,
                            num_layers=1, bidirectional=True)
        self.att_weight = nn.Parameter(torch.randn(config.bi_lstm_hidden, config.batch_size, config.bi_lstm_hidden))
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.nlayers)
        if config.is_pretrained_model:
            # with torch.no_grad():
            config_bert = BertConfig.from_pretrained(config.pretrained_config)
            model = BertModel.from_pretrained(config.pretrained_model, config=config_bert)
            self.embedding = model
            for name, param in model.named_parameters():
                param.requires_grad = True
        elif config.is_vector:
            self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
        self.embedding.weight.requires_grad = True
        self.emsize = config.embedding_size
        self.linner = nn.Linear(config.bi_lstm_hidden, ntag)
        self.init_weights()
        self.crflayer = CRF(ntag)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_hidden_lstm(self):
        return (torch.randn(2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(device),
                torch.randn(2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(device))

    def init_weights(self):
        initrange = 0.1
        self.linner.bias.data.zero_()
        self.linner.weight.data.uniform_(-initrange, initrange)

    def _get_src_key_padding_mask(self, text_len, seq_len):
        batchszie = text_len.size(0)
        list1 = []
        for i in range(batchszie):
            list2 = []
            list2.append([False for i in range(text_len[i])] + [True for i in range(seq_len - text_len[i])])
            list1.append(list2)
        src_key_padding_mask = torch.tensor(np.array(list1)).squeeze(1)
        return src_key_padding_mask

    def loss(self, src, text_len, tag):
        mask_crf = torch.ne(src, 1)
        transformer_out = self.transformer_forward(src, text_len)
        lstm_out, _ = self.lstm(transformer_out)
        emissions = self.linner(lstm_out)
        crf_loss = self.crflayer(emissions, tag, mask=mask_crf) / tag.size(1)
        return -crf_loss

    def forward(self, src, text_len):
        # self.hidden = self.init_hidden_lstm()
        mask_crf = torch.ne(src, 1)
        transformer_out = self.transformer_forward(src, text_len)
        lstm_out, self.hidden = self.lstm(transformer_out)
        emissions = self.linner(lstm_out)
        return self.crflayer.decode(emissions, mask=mask_crf)

    def transformer_forward(self, src, text_len):
        src_key_padding_mask = self._get_src_key_padding_mask(text_len, src.size(0))
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask
        src = self.embedding(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=self.src_mask.to(device),
                                          src_key_padding_mask=src_key_padding_mask.to(device))
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BiLSTM_CRF(nn.Module):
    def __init__(self, config, ntoken, ntag, vectors):
        super(BiLSTM_CRF, self).__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.embedding = nn.Embedding(ntoken, config.embedding_size)
        if config.is_vector:
            self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
        self.lstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.bi_lstm_hidden // 2,
                            num_layers=config.num_layers, bidirectional=True)
        self.linner = nn.Linear(config.bi_lstm_hidden, ntag)
        self.crflayer = CRF(ntag)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.config.num_layers * 2, batch_size, self.config.bi_lstm_hidden // 2).to(device)
        c0 = torch.zeros(self.config.num_layers * 2, batch_size, self.config.bi_lstm_hidden // 2).to(device)

        return h0, c0

    def forward(self, text, lengths):
        mask = torch.ne(text, self.config.pad_index)
        emission = self.lstm_forward(text, lengths)
        return self.crflayer.decode(emission, mask)

    def loss(self, text, lengths, tag):
        mask = torch.ne(text, self.config.pad_index)
        emission = self.lstm_forward(text, lengths)
        # crf_loss = -self.crflayer(emission, tag, mask=mask) / tag.size(1)
        crf_loss = -self.crflayer(emission, tag, mask=mask)
        return crf_loss

    def lstm_forward(self, text, lengths):
        text = self.embedding(text)
        text = pack_padded_sequence(text, lengths)
        self.hidden = self.init_hidden(batch_size=len(lengths))
        lstm_out, self.hidden = self.lstm(text, self.hidden)
        lstm_out, new_lengths = pad_packed_sequence(lstm_out)
        emission = self.linner(lstm_out)
        return emission

class BiLSTM_CRF_changed(nn.Module):
    def __init__(self, config, ntoken, ntag):
        super(BiLSTM_CRF_changed, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(ntoken, config.embedding_size)
        self.lstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.bi_lstm_hidden // 2,
                            num_layers=config.num_layers, bidirectional=True)
        self.linner = nn.Linear(config.bi_lstm_hidden, ntag)
        self.crflayer = CRF(ntag)

    def init_hidden(self):
        h0 = torch.zeros(self.config.num_layers * 2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(device)
        c0 = torch.zeros(self.config.num_layers * 2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(device)
        return h0, c0

    def forward(self, texts, lengths):
        lengths_mask = [sum(lengths[i]) for i in range(lengths.size(0))]
        outputs = []
        for i in range(texts.size(1)):
            outputs.append(self.lstm_forward(texts[:, i:i + 1, :].squeeze(1), lengths[:, i:i + 1].squeeze(1)))
        lstm1_output = torch.cat(outputs, dim=0)
        lstm2_input, lengths = self.get_text_lengths(lstm1_output, lengths, size=texts.size(2))
        max_len = max(lengths)
        lstm2_input = pack_padded_sequence(lstm2_input, lengths, enforce_sorted=False)
        lstm2_output = self.lstm(lstm2_input, self.hidden)[0]
        lstm2_output, _ = pad_packed_sequence(lstm2_output)
        emission = self.linner(lstm2_output)
        mask = []
        for i in range(len(lengths_mask)):
            mask.append([])
            for j in range(lengths_mask[i]):
                mask[i].append(True)
            for j in range(max_len - lengths_mask[i]):
                mask[i].append(False)
        mask = torch.tensor(np.array(mask)).permute(1, 0).to(device)
        return self.crflayer.decode(emission, mask)

    def loss(self, texts, lengths, tag):
        lengths_mask = [sum(lengths[i]) for i in range(lengths.size(0))]
        outputs = []
        for i in range(texts.size(1)):
            outputs.append(self.lstm_forward(texts[:,i:i+1,:].squeeze(1), lengths[:,i:i+1].squeeze(1)))
        lstm1_output = torch.cat(outputs, dim=0)
        lstm2_input, lengths = self.get_text_lengths(lstm1_output, lengths, size=texts.size(2))
        max_len = max(lengths)
        lstm2_input = pack_padded_sequence(lstm2_input, lengths, enforce_sorted=False)
        lstm2_output = self.lstm(lstm2_input, self.hidden)[0]
        lstm2_output, _ = pad_packed_sequence(lstm2_output)
        emission = self.linner(lstm2_output)
        tag = tag.permute(1,0)[:max_len,:]
        mask = []
        for i in range(len(lengths_mask)):
            mask.append([])
            for j in range(lengths_mask[i]):
                mask[i].append(True)
            for j in range(max_len-lengths_mask[i]):
                mask[i].append(False)
        mask = torch.tensor(np.array(mask)).permute(1,0).to(device)
        return -self.crflayer(emission, tag, mask)

    def lstm_forward(self, text, lengths):
        text = self.embedding(text).permute(1,0,2)
        if 0 in lengths.cpu().numpy().tolist():
            start = lengths.cpu().numpy().tolist().index(0)
            before_lengths = lengths[:start]
            max_len = max(before_lengths.cpu().numpy().tolist())
            before_text = text[:,:start,:]
            before_pad_text = before_text[max_len:,:,:]
            after_text = text[:,start:,:]
            text = pack_padded_sequence(before_text, before_lengths, enforce_sorted=False)
            self.hidden = self.init_hidden()
            lstm_out, self.hidden = self.lstm(text, self.hidden)
            lstm_out, new_lengths = pad_packed_sequence(lstm_out)
            lstm_out = torch.cat([lstm_out, before_pad_text], dim=0)
            lstm_out = torch.cat([lstm_out, after_text], dim=1)
        else:
            max_len = max(lengths.cpu().numpy().tolist())
            pad_text = text[max_len:,:,:]
            text = pack_padded_sequence(text, lengths, enforce_sorted=False)
            self.hidden = self.init_hidden()
            lstm_out, self.hidden = self.lstm(text)
            lstm_out, new_lengths = pad_packed_sequence(lstm_out)
            lstm_out = torch.cat([lstm_out, pad_text],dim=0)
        return lstm_out

    def get_text_lengths(self, lstm1_output, lengths, size):
        list1 = []
        list2 = []
        list3 = []
        lengths = lengths.cpu().numpy().tolist()
        for i in range(len(lengths)):
            output_i = lstm1_output[:,i:i+1,:].squeeze(1)
            for j in range(len(lengths[i])):
                start = j*size
                middle = start + lengths[i][j]
                end = (j+1)*size
                list1.append(output_i[start:middle,:])
                list2.append(output_i[middle:end,:])
            list3.append(torch.cat([torch.cat(list1,dim=0), torch.cat(list2,dim=0)], dim=0).unsqueeze(0))
            list1 = []
            list2 = []
        tensor = torch.cat(list3, dim=0).permute(1,0,2)
        list3=[]
        new_lengths = torch.tensor(np.array([sum(l) for l in lengths], dtype=np.int64))
        return tensor, new_lengths

class BiLSTM_CRF_ATT(BiLSTM_CRF):
    def __init__(self, config, ntoken, ntag, vectors):
        super(BiLSTM_CRF_ATT,self).__init__(config, ntoken, ntag, vectors) # 继承父类的属性
        self.attention = MultiHeadAttention(self.config.key_dim, self.config.val_dim, self.config.bi_lstm_hidden,
                        self.config.num_heads, self.config.attn_dropout)

    def lstm_forward(self, text, lengths):
        text = self.embedding(text)
        text = pack_padded_sequence(text, lengths)
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(text, self.hidden)
        lstm_out, new_lengths = pad_packed_sequence(lstm_out)
        output = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(output, output, output, None)
        attn_out = attn_out.permute(1, 0, 2)
        return self.linner(attn_out)

class CNN_CRF(nn.Module):
    def __init__(self, config, ntoken, ntag):
        super(CNN_CRF, self).__init__()
        self.config = config
        self.sizes = [3,5,7]
        self.embedding = nn.Embedding(ntoken, config.embedding_size)
        if config.is_vector:
            vectors = Vectors(name='./vector/sgns.wiki.word')
            self.embedding = nn.Embedding.from_pretrained(vectors)
        self.convs = nn.ModuleList([nn.Conv2d(config.chanel_num, config.filter_num, (size, config.embedding_size), padding=size//2) for size in self.sizes])
        self.linner = nn.Linear(config.bi_lstm_hidden, ntag)
        self.crflayer = CRF(ntag)

    def loss(self, text, lengths, tag):
        mask = torch.ne(text, self.config.pad_index)
        emission = self.cnn_forward(text)
        return self.crflayer(emission, tag, mask)

    def forwarf(self, text, lengths):
        mask = torch.ne(text, self.config.pad_index)
        emission = self.cnn_forward(text)
        return self.crflayer.decode(emission, mask)

    def cnn_forward(self, text):
        text = self.embedding(text).transpose(0, 1).unsqueeze(1)
        cnn_out = [F.relu(conv(text)) for conv in self.convs]
        for i in range(len(self.sizes)):
            x = int((self.sizes[i] - 1) / 2)
            cnn_out[i] = cnn_out[i][:, :, :, x]
        return self.linner(torch.cat(cnn_out, 1).squeeze().permute(2, 0, 1))

class BiLSTM_CRF_DAE(nn.Module):
    def __init__(self, config, ntoken, ntag, vectors):
        super(BiLSTM_CRF_DAE, self).__init__()
        self.config = config
        self.vocab_size = ntoken
        self.batch_size = config.batch_size
        self.dropout = config.dropout
        self.drop = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(ntoken, config.embedding_size)
        if config.is_vector:
            self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
        self.lstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.bi_lstm_hidden // 2,
                            num_layers=config.num_layers, bidirectional=True)
        self.linner = nn.Linear(config.bi_lstm_hidden, ntag)
        self.lm_decoder = nn.Linear(config.bi_lstm_hidden, self.vocab_size)
        self.dice_loss = DiceLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.crflayer = CRF(ntag)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.config.num_layers * 2, batch_size, self.config.bi_lstm_hidden // 2).to(device)
        c0 = torch.zeros(self.config.num_layers * 2, batch_size, self.config.bi_lstm_hidden // 2).to(device)

        return h0, c0

    def loss(self, x, sent_lengths, y):
        mask = torch.ne(x, self.config.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        crf_loss = -self.crflayer(emissions, y, mask=mask) / y.size(1)
        src_encoding = self.encode(x, sent_lengths)
        lm_output = self.decode_lm(src_encoding)
        lm_loss = self.criterion(lm_output.view(-1, self.vocab_size), x.view(-1))
        return crf_loss + lm_loss


    def forward(self, x, sent_lengths):
        mask = torch.ne(x, self.config.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        return self.crflayer.decode(emissions, mask=mask)

    def lstm_forward(self, sentence, sent_lengths):
        x = self.embedding(sentence.to(device)).to(device)
        x = pack_padded_sequence(x, sent_lengths)
        self.hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size.to(device))
        y = self.linner(lstm_out.to(device))
        return y.to(device)

    def encode(self, source, length):
        embed = self.embedding(source)
        packed_src_embed = pack_padded_sequence(embed, length)
        _, hidden = self.lstm(packed_src_embed)
        embed = self.drop(self.embedding(source))
        packed_src_embed = pack_padded_sequence(embed, length)
        lstm_output, _ = self.lstm(packed_src_embed, hidden)
        lstm_output = pad_packed_sequence(lstm_output)
        lstm_output = self.drop(lstm_output[0])
        return lstm_output

    def decode_lm(self, src_encoding):
        decoded = self.lm_decoder(
            src_encoding.contiguous().view(src_encoding.size(0) * src_encoding.size(1), src_encoding.size(2)))
        lm_output = decoded.view(src_encoding.size(0), src_encoding.size(1), decoded.size(1))
        return lm_output

class TransformerEncoderModel_DAE(nn.Module):
    def __init__(self, config, ntoken, ntag, vectors):
        super(TransformerEncoderModel_DAE, self).__init__()
        self.config = config
        self.src_mask = None
        self.vectors = vectors
        self.vocab_size = ntoken
        self.embedding = nn.Embedding(ntoken, config.embedding_size)
        self.pos_encoder = PositionalEncoding(config.embedding_size, config.dropout)
        encoder_layers = TransformerEncoderLayer(config.embedding_size, config.nhead, config.nhid, config.dropout)
        self.lstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.bi_lstm_hidden // 2,
                            num_layers=1, bidirectional=True)
        self.att_weight = nn.Parameter(torch.randn(config.bi_lstm_hidden, config.batch_size, config.bi_lstm_hidden))
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.nlayers)
        if config.is_pretrained_model:
            # with torch.no_grad():
            config_bert = BertConfig.from_pretrained(config.pretrained_config)
            model = BertModel.from_pretrained(config.pretrained_model, config=config_bert)
            self.embedding = model
            for name, param in model.named_parameters():
                param.requires_grad = True
        elif config.is_vector:
            self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
        self.embedding.weight.requires_grad = True
        self.emsize = config.embedding_size
        self.linner = nn.Linear(config.bi_lstm_hidden, ntag)
        self.init_weights()
        self.crflayer = CRF(ntag)
        self.lm_decoder = nn.Linear(config.bi_lstm_hidden, self.vocab_size)
        self.dice_loss = DiceLoss()
        self.criterion = nn.CrossEntropyLoss()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_hidden_lstm(self):
        return (torch.randn(2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(device),
                torch.randn(2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(device))

    def init_weights(self):
        initrange = 0.1
        self.linner.bias.data.zero_()
        self.linner.weight.data.uniform_(-initrange, initrange)

    def _get_src_key_padding_mask(self, text_len, seq_len):
        batchszie = text_len.size(0)
        list1 = []
        for i in range(batchszie):
            list2 = []
            list2.append([False for i in range(text_len[i])] + [True for i in range(seq_len - text_len[i])])
            list1.append(list2)
        src_key_padding_mask = torch.tensor(np.array(list1)).squeeze(1)
        return src_key_padding_mask

    def loss(self, src, text_len, tag, weight=6.4):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        mask_crf = torch.ne(src, 1)
        transformer_out = self.transformer_forward(src, text_len)
        src_encoding = self.encode(transformer_out)
        lm_output = self.decode_lm(src_encoding)
        lm_loss = w * self.criterion(lm_output.view(-1, self.vocab_size), src.view(-1))
        lstm_out, _ = self.lstm(transformer_out)
        emissions = self.linner(lstm_out)
        crf_loss = -self.crflayer(emissions, tag, mask=mask_crf) / tag.size(1)
        dice_loss = self.dice_loss(emissions, tag).to(device)
        # att_out = torch.bmm(lstm_out.transpose(0,1), self.att_weight.transpose(0,1)).transpose(0,1)
        return crf_loss + lm_loss

    def forward(self, src, text_len):
        mask_crf = torch.ne(src, 1)
        transformer_out = self.transformer_forward(src, text_len)
        lstm_out, _ = self.lstm(transformer_out)
        # att_out = torch.bmm(lstm_out.transpose(0,1), self.att_weight.transpose(0,1)).transpose(0,1)
        emissions = self.linner(lstm_out)
        return self.crflayer.decode(emissions, mask=mask_crf)

    def transformer_forward(self, src, text_len):
        src_key_padding_mask = self._get_src_key_padding_mask(text_len, src.size(0))
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask
        # Transformer
        # src = self.embedding(src)[0]
        src = self.embedding(src) * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=self.src_mask.to(device),
                                          src_key_padding_mask=src_key_padding_mask.to(device))
        return output

    def encode(self, source):
        # embed = self.embedding(source)
        _, hidden = self.lstm(source)
        output, _ = self.lstm(source, hidden)
        return output

    def decode_lm(self, src_encoding):
        decoded = self.lm_decoder(
            src_encoding.contiguous().view(src_encoding.size(0) * src_encoding.size(1), src_encoding.size(2)))
        lm_output = decoded.view(src_encoding.size(0), src_encoding.size(1), decoded.size(1))
        return lm_output