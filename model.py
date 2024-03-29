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
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base.layers import ScaledDotProductAttention, SelfAttention, MultiHeadAttention


class TransformerEncoderModel(nn.Module):
    def __init__(self, config, ntoken, ntag, vectors):
        super(TransformerEncoderModel, self).__init__()
        self.ntoken = ntoken
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
        self.dice_loss = DiceLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.lm_decoder = nn.Linear(self.config.bi_lstm_hidden, ntoken)

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
        # crf_loss = self.crflayer(emissions, tag, mask=mask_crf) / tag.size(1)
        dice = self.config.dice_loss_weight * self.dice_loss(emissions.permute(1, 2, 0), tag.permute(1, 0))
        if self.config.is_dice_loss:
            crf_loss = self.crflayer(emissions, tag, mask=mask_crf) - self.config.dice_loss_weight * self.dice_loss(
                emissions.permute(1, 2, 0), tag.permute(1, 0))
        elif self.config.is_dae_loss:
            a = self.config.daen_loss_weight * self.dae_loss(src, text_len)
            crf_loss = self.crflayer(emissions, tag, mask=mask_crf) - self.config.dae_loss_weight * self.dae_loss(src,
                                                                                                                  text_len)
        else:
            crf_loss = self.crflayer(emissions, tag, mask=mask_crf)
        return -crf_loss, dice

    def forward(self, src, text_len):
        self.hidden = self.init_hidden_lstm()
        mask_crf = torch.ne(src, 1)
        transformer_out = self.transformer_forward(src, text_len)
        lstm_out, _ = self.lstm(transformer_out, self.hidden)
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

    def dae_loss(self, src, text_len):
        src_encoding = self.encode(src, text_len)
        lm_output = self.decode_lm(src_encoding)
        lm_loss = self.criterion(lm_output.view(-1, self.ntoken), src.view(-1))
        return lm_loss

    def encode(self, source, length):
        # _, hidden = self.lstm(source)
        # output, _ = self.lstm(source, hidden)
        embed = self.embedding(source)
        packed_src_embed = pack_padded_sequence(embed, length)
        _, hidden = self.lstm(packed_src_embed)
        # embed = self.drop(self.embedding(source))
        packed_src_embed = pack_padded_sequence(embed, length)
        lstm_output, _ = self.lstm(packed_src_embed, hidden)
        lstm_output = pad_packed_sequence(lstm_output)
        # lstm_output = self.drop(lstm_output[0])
        return lstm_output[0]

    def decode_lm(self, src_encoding):
        decoded = self.lm_decoder(
            src_encoding.contiguous().view(src_encoding.size(0) * src_encoding.size(1), src_encoding.size(2)))
        lm_output = decoded.view(src_encoding.size(0), src_encoding.size(1), decoded.size(1))
        return lm_output


class FLAT(nn.Module):
    def __init__(self, config, nbigram, nlattice, ntag, vectors):
        super(FLAT, self).__init__()
        self.config = config
        self.src_mask = None
        self.vectors = vectors
        self.embedding_size = config.embedding_size
        self.bigram_embedding_size = config.bigram_embedding_size
        self.lattice_embedding_size = config.lattice_embedding_size
        self.bigram_embedding = nn.Embedding(nbigram, config.bigram_embedding_size)
        self.lattice_embedding = nn.Embedding(nlattice, config.lattice_embedding_size)
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
        self.emsize = config.embedding_size
        self.linner = nn.Linear(config.bi_lstm_hidden, ntag)
        self.lattice_linner = nn.Linear(config.lattice_embedding_size, config.bi_lstm_hidden)
        self.big_lat_linner = nn.Linear(config.bi_lstm_hidden, config.bi_lstm_hidden)
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

    def loss(self, bigram, lattice, lattice_len, tag):
        mask_crf = torch.ne(bigram, 1)
        transformer_out = self.transformer_forward(bigram, lattice, lattice_len)[0:bigram.size(0), :, :]
        lstm_out, _ = self.lstm(transformer_out)
        emissions = self.linner(lstm_out)
        crf_loss = self.crflayer(emissions, tag, mask=mask_crf)
        return -crf_loss

    def forward(self, bigram, lattice, lattice_len):
        mask_crf = torch.ne(bigram, 1)
        transformer_out = self.transformer_forward(bigram, lattice, lattice_len)[0:bigram.size(0), :, :]
        lstm_out, self.hidden = self.lstm(transformer_out)
        emissions = self.linner(lstm_out)
        return self.crflayer.decode(emissions, mask=mask_crf)

    def transformer_forward(self, bigram, lattice, lattice_len):
        src_key_padding_mask = self._get_src_key_padding_mask(lattice_len, lattice.size(0))
        if self.src_mask is None or self.src_mask.size(0) != len(lattice):
            mask = self._generate_square_subsequent_mask(len(lattice))
            self.src_mask = mask
        bigram_embedding = self.bigram_embedding(bigram) * math.sqrt(self.embedding_size)
        lattice_embedding = self.lattice_embedding(lattice) * math.sqrt(self.embedding_size)
        x = torch.zeros(size=[lattice_embedding.size(0) - bigram_embedding.size(0), lattice_embedding.size(1),
                              lattice_embedding.size(2)]).to(device)
        bigram_embedding = torch.cat([bigram_embedding, x], dim=0)
        big_lat_embedding = self.big_lat_linner(torch.cat([bigram_embedding, lattice_embedding], dim=-1))
        lattice_embedding = self.lattice_linner(lattice_embedding)
        src = (big_lat_embedding + lattice_embedding) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=self.src_mask.to(device),
                                          src_key_padding_mask=src_key_padding_mask.to(device))
        return output


class CNN_TransformerEncoderModel(nn.Module):
    def __init__(self, config, ntoken, ntag, vectors):
        super(CNN_TransformerEncoderModel, self).__init__()
        self.config = config
        self.src_mask = None
        self.vectors = vectors

        self.sizes = [3, 5, 7]
        if config.is_vector:
            vectors = Vectors(name='./vector/sgns.wiki.word')
            self.embedding = nn.Embedding.from_pretrained(vectors)
        self.convs = nn.ModuleList(
            [nn.Conv2d(config.chanel_num, config.filter_num, (size, config.embedding_size), padding=size // 2) for size
             in self.sizes])

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
        cnn_input = src.transpose(0, 1).unsqueeze(1)

        cnn_out = [F.relu(conv(cnn_input)) for conv in self.convs]
        for i in range(len(self.sizes)):
            x = int((self.sizes[i] - 1) / 2)
            cnn_out[i] = cnn_out[i][:, :, :, x]
        cnn_out = torch.cat(cnn_out, 1).permute(2, 0, 1)
        src = self.pos_encoder(cnn_out)  # 64 100 6

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
        emission, _ = self.lstm_forward(text, lengths)
        return self.crflayer.decode(emission, mask)

    def loss(self, text, lengths, tag):
        mask = torch.ne(text, self.config.pad_index)
        emission, _ = self.lstm_forward(text, lengths)
        crf_loss = -self.crflayer(emission, tag, mask=mask)
        return crf_loss

    def lstm_forward(self, text, lengths):
        text = self.embedding(text)
        text = pack_padded_sequence(text, lengths)
        hidden = self.init_hidden(batch_size=len(lengths))
        lstm_out, new_hidden = self.lstm(text, hidden)
        lstm_out, new_lengths = pad_packed_sequence(lstm_out)
        emission = self.linner(lstm_out)
        return emission, new_hidden[0].view(self.config.batch_size, -1)


class BiLSTM_CRF_hidden_tag(BiLSTM_CRF):
    def __init__(self, config, ntoken, ntag, hidden_ntag, vectors):
        super(BiLSTM_CRF_hidden_tag, self).__init__(config, ntoken, ntag, vectors)
        self.linner_hidden_tag = nn.Linear(config.bi_lstm_hidden, hidden_ntag)
        self.crflayer_hidden_tag = CRF(hidden_ntag)

    def loss(self, text, lengths, tag, hidden_tag):
        mask = torch.ne(text, self.config.pad_index)
        emission, new_hidden = self.lstm_forward(text, lengths)
        emission_hidden_tag = self.linner_hidden_tag(new_hidden).unsqueeze(1)
        hidden_tag_loss = -self.crflayer_hidden_tag(emission_hidden_tag, hidden_tag.permute(1, 0))
        crf_loss = -self.crflayer(emission, tag, mask=mask)
        return crf_loss + self.config.weight * hidden_tag_loss


class BiLSTM_CRF_changed(nn.Module):
    def __init__(self, config, ntoken, ntag):
        super(BiLSTM_CRF_changed, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(ntoken, config.embedding_size)
        self.lstm1 = nn.LSTM(input_size=config.embedding_size, hidden_size=config.bi_lstm_hidden // 2,
                             num_layers=config.num_layers, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=config.embedding_size, hidden_size=config.bi_lstm_hidden // 2,
                             num_layers=config.num_layers, bidirectional=True)
        self.linner1 = nn.Linear(config.bi_lstm_hidden, ntag)
        self.linner2 = nn.Linear(config.bi_lstm_hidden, 2)
        self.crflayer1 = CRF(ntag)
        self.crflayer2 = CRF(2)

    def init_hidden(self):
        h0 = torch.zeros(self.config.num_layers * 2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(device)
        c0 = torch.zeros(self.config.num_layers * 2, self.config.batch_size, self.config.bi_lstm_hidden // 2).to(device)
        return h0, c0

    def forward(self, texts, lengths):
        lengths_mask = [sum(lengths[i]) for i in range(lengths.size(0))]
        results = [[] for i in range(texts.size(0))]
        hidden = self.init_hidden()
        for i in range(texts.size(1)):
            output, hidden, result = self.lstm_forward(texts[:, i:i + 1, :].squeeze(1), lengths[:, i:i + 1].squeeze(1),
                                                       None, hidden, None)
            if result is not None:
                for i in range(len(result)):
                    results[i].extend(result[i])
        # lstm1_output = torch.cat(outputs, dim=0)
        # lstm2_input, lengths = self.get_text_lengths(lstm1_output, lengths, size=texts.size(2))
        # max_len = max(lengths)
        # 前两层
        # lstm2_input = pack_padded_sequence(lstm2_input, lengths, enforce_sorted=False)
        # hidden = self.init_hidden()
        # lstm2_output = self.lstm2(lstm2_input, hidden)[0]
        # lstm2_output, _ = pad_packed_sequence(lstm2_output)
        # 第一层
        # lstm2_output = lstm2_input[:max_len, :, :]
        # emission = self.linner(lstm2_output)
        # mask = self.get_mask(lengths_mask, max_len)
        # return self.crflayer2.decode(emission, mask)
        return results

    def loss(self, texts, lengths, tag, sub_tag, hidden_tag):
        lengths_mask = [sum(lengths[i]) for i in range(lengths.size(0))]
        outputs = []
        losses = []
        hidden = self.init_hidden()
        for i in range(texts.size(1)):
            x1, x2, hidden = self.lstm_forward(texts[:, i:i + 1, :].squeeze(1), lengths[:, i:i + 1].squeeze(1),
                                               sub_tag[:, i:i + 1, :].squeeze(1), hidden,
                                               hidden_tag[:, i:i + 1].squeeze(1))
            outputs.append(x1)
            losses.append(x2)
        # lstm1_output = torch.cat(outputs, dim=0)
        # lstm2_input, lengths = self.get_text_lengths(lstm1_output, lengths, size=texts.size(2))
        # max_len = max(lengths)
        # # 前两层
        # lstm2_input = pack_padded_sequence(lstm2_input, lengths, enforce_sorted=False)
        # hidden = self.init_hidden()
        # lstm2_output, _ = self.lstm2(lstm2_input, hidden)
        # lstm2_output, _ = pad_packed_sequence(lstm2_output)
        # # 第一层
        # # lstm2_output = lstm2_input[:max_len,:,:]
        # emission = self.linner(lstm2_output)
        # tag = tag.permute(1,0)[:max_len,:]
        # mask = self.get_mask(lengths_mask, max_len)
        return sum(losses)

    def lstm_forward(self, text, lengths, sub_tag, hidden, hidden_tag):
        text = self.embedding(text).permute(1, 0, 2)
        if sub_tag is not None:
            sub_tag = sub_tag.permute(1, 0)
        loss1 = None
        loss2 = None
        emission1 = None
        mask1 = None
        if 0 in lengths.cpu().numpy().tolist():
            start = lengths.cpu().numpy().tolist().index(0)
            before_lengths = lengths[:start]
            if len(before_lengths) == 0:
                lstm_out = text
                new_hidden = hidden
            else:
                max_len = max(before_lengths.cpu().numpy().tolist())
                before_text = text[:, :start, :]
                before_pad_text = before_text[max_len:, :, :]
                after_text = text[:, start:, :]
                text = pack_padded_sequence(before_text, before_lengths, enforce_sorted=False)
                lstm_out, new_hidden = self.lstm1(text, hidden)
                lstm_out, new_lengths = pad_packed_sequence(lstm_out)
                lstm_out = torch.cat([lstm_out, before_pad_text], dim=0)
                emission1 = self.linner1(lstm_out)
                mask1 = []
                for i in range(len(before_lengths)):
                    mask1.append([])
                    for j in range(before_lengths[i]):
                        mask1[i].append(True)
                    for j in range(emission1.size(0) - before_lengths[i]):
                        mask1[i].append(False)
                mask1 = torch.tensor(np.array(mask1)).permute(1, 0).to(device)
                if sub_tag is not None:
                    sub_tag = sub_tag[:, :start]
                    loss1 = -self.crflayer1(emission1, sub_tag, mask1)
                    emission2 = self.linner2(new_hidden[0].view(before_lengths.size(0), -1)).unsqueeze(1)
                    hidden_tag = hidden_tag[:len(before_lengths)].unsqueeze(1)
                    loss2 = -self.crflayer2(emission2, hidden_tag)
                lstm_out = torch.cat([lstm_out, after_text], dim=1)
        else:
            max_len = max(lengths.cpu().numpy().tolist())
            pad_text = text[max_len:, :, :]
            text = pack_padded_sequence(text, lengths, enforce_sorted=False)
            lstm_out, new_hidden = self.lstm1(text, hidden)
            lstm_out, new_lengths = pad_packed_sequence(lstm_out)
            lstm_out = torch.cat([lstm_out, pad_text], dim=0)
            emission1 = self.linner1(lstm_out)
            mask1 = []
            for i in range(len(lengths)):
                mask1.append([])
                for j in range(lengths[i]):
                    mask1[i].append(True)
                for j in range(emission1.size(0) - lengths[i]):
                    mask1[i].append(False)
            mask1 = torch.tensor(np.array(mask1)).permute(1, 0).to(device)
            if sub_tag is not None:
                loss1 = -self.crflayer1(emission1, sub_tag, mask1)
                emission2 = self.linner2(new_hidden[0].view(self.config.batch_size, -1)).unsqueeze(1)
                hidden_tag = hidden_tag.unsqueeze(1)
                loss2 = -self.crflayer2(emission2, hidden_tag)
        if sub_tag is not None:
            return lstm_out, loss1 + loss2, new_hidden
        elif emission1 is not None:
            return lstm_out, new_hidden, self.crflayer1.decode(emission1, mask1)
        elif emission1 is None:
            return lstm_out, new_hidden, None

    def get_mask(self, lengths_mask, max_len):
        mask = []
        for i in range(len(lengths_mask)):
            mask.append([])
            for j in range(lengths_mask[i]):
                mask[i].append(True)
            for j in range(max_len - lengths_mask[i]):
                mask[i].append(False)
        mask = torch.tensor(np.array(mask)).permute(1, 0).to(device)
        return mask

    def get_text_lengths(self, lstm1_output, lengths, size):
        list1 = []
        list2 = []
        list3 = []
        lengths = lengths.cpu().numpy().tolist()
        for i in range(len(lengths)):
            output_i = lstm1_output[:, i:i + 1, :].squeeze(1)
            for j in range(len(lengths[i])):
                start = j * size
                middle = start + lengths[i][j]
                end = (j + 1) * size
                list1.append(output_i[start:middle, :])
                list2.append(output_i[middle:end, :])
            list3.append(torch.cat([torch.cat(list1, dim=0), torch.cat(list2, dim=0)], dim=0).unsqueeze(0))
            list1 = []
            list2 = []
        tensor = torch.cat(list3, dim=0).permute(1, 0, 2)
        list3 = []
        new_lengths = torch.tensor(np.array([sum(l) for l in lengths], dtype=np.int64))
        return tensor, new_lengths


class BiLSTM_CRF_ATT(BiLSTM_CRF):
    def __init__(self, config, ntoken, ntag, vectors):
        super(BiLSTM_CRF_ATT, self).__init__(config, ntoken, ntag, vectors)  # 继承父类的属性
        self.attention = MultiHeadAttention(self.config.key_dim, self.config.val_dim, self.config.bi_lstm_hidden,
                                            self.config.num_heads, self.config.attn_dropout)

    def lstm_forward(self, text, lengths):
        text = self.embedding(text)
        text = pack_padded_sequence(text, lengths)
        hidden = self.init_hidden()
        lstm_out, new_hidden = self.lstm(text, hidden)
        lstm_out, new_lengths = pad_packed_sequence(lstm_out)
        output = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(output, output, output, None)
        attn_out = attn_out.permute(1, 0, 2)
        return self.linner(attn_out), new_hidden[0].view(self.config.batch_size, -1)


class CNN_CRF(nn.Module):
    def __init__(self, config, ntoken, ntag):
        super(CNN_CRF, self).__init__()
        self.config = config
        self.sizes = [3, 5, 7]
        self.embedding = nn.Embedding(ntoken, config.embedding_size)
        if config.is_vector:
            vectors = Vectors(name='./vector/sgns.wiki.word')
            self.embedding = nn.Embedding.from_pretrained(vectors)
        self.convs = nn.ModuleList(
            [nn.Conv2d(config.chanel_num, config.filter_num, (size, config.embedding_size), padding=size // 2) for size
             in self.sizes])
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


class CNN_BiLSTM_CRF(BiLSTM_CRF):
    def __init__(self, config, ntoken, ntag, vectors):
        super(CNN_BiLSTM_CRF, self).__init__(config, ntoken, ntag, vectors)  # 继承父类的属性
        self.sizes = [3, 5, 7]
        self.convs = nn.ModuleList(
            [nn.Conv2d(config.chanel_num, config.filter_num, (size, config.embedding_size), padding=size // 2) for size
             in self.sizes])

    def loss(self, text, lengths, tag):
        mask = torch.ne(text, self.config.pad_index)
        hidden = text
        for i in range(3):
            hidden = self.cnn_forward(hidden, i)
        emission, _ = self.lstm_forward(hidden, lengths)
        return -self.crflayer(emission, tag, mask)

    def forward(self, text, lengths):
        mask = torch.ne(text, self.config.pad_index)
        hidden = text
        for i in range(2):
            hidden = self.cnn_forward(hidden, i)
        emission, _ = self.lstm_forward(hidden, lengths)
        return self.crflayer.decode(emission, mask)

    def cnn_forward(self, text, i):
        if i == 0:
            text = self.embedding(text).transpose(0, 1).unsqueeze(1)
        else:
            text = text.transpose(0, 1).unsqueeze(1)
        cnn_out = [F.relu(conv(text)) for conv in self.convs]
        for i in range(len(self.sizes)):
            x = int((self.sizes[i] - 1) / 2)
            cnn_out[i] = cnn_out[i][:, :, :, x]
        return torch.cat(cnn_out, 1).squeeze().permute(2, 0, 1)

    def lstm_forward(self, text, lengths):
        hidden = self.init_hidden(batch_size=len(lengths))
        lstm_out, new_hidden = self.lstm(text, hidden)
        emission = self.linner(lstm_out)
        return emission, new_hidden[0].view(self.config.batch_size, -1)


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
        crf_loss = -self.crflayer(emissions, y, mask=mask)
        src_encoding = self.encode(x, sent_lengths)
        lm_output = self.decode_lm(src_encoding)
        lm_loss = self.criterion(lm_output.view(-1, self.vocab_size), x.view(-1))
        return crf_loss + self.config.weight * lm_loss

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
        w = torch.cuda.FloatTensor(1).fill_(weight).to(device)
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
        cnn_input = src.transpose(0, 1).unsqueeze(1)

        cnn_out = [F.relu(conv(cnn_input)) for conv in self.convs]
        for i in range(len(self.sizes)):
            x = int((self.sizes[i] - 1) / 2)
            cnn_out[i] = cnn_out[i][:, :, :, x]
        cnn_out = torch.cat(cnn_out, 1).permute(2, 0, 1)
        src = self.pos_encoder(cnn_out)  # 64 100 6

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
