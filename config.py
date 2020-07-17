# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/4/26 10:25

import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

default_config = {
    'experiment_name' : 'BiLSTM_CRF_changed_firstLevel', # 实验名称
    'model_path': './save_model/{}.pkl',
    'analysis_path' : './result/data/{}/analysis.xlsx',
    'train_path' : './data/sub_train.xlsx',
    'dev_path' : './data/sub_dev.xlsx',
    'train_dev_path' : './task2_train_reformat_cleaned.xlsx',
    'test_path' : './data/task2_no_val_cleaned.xlsx',
    'unformated_val_path' : './result/data/{}/unformated_val.xlsx',  #模型训练直接预测
    'test_formated_val_path' : './result/data/{}/test_format/formated_val.xlsx',    #测试format结果是否有提升
    'test_unformated_val_path' : './result/data/{}/test_format/unformated_val.xlsx',
    'model_name' : 'BiLSTM_CRF_changed', # TransformerEncoderModel_DAE or BiLSTM_CRF or
    # TransformerEncoderModel or CNN_CRF or BiLSTM_CRF_ATT or BiLSTM_CRF_changed
    'is_pretrained_model' : False,
    'pretrained_config' : './pretrained_models/RoBERTa/config.json',
    'pretrained_model' : './pretrained_models/RoBERTa/pytorch_model.bin',
    'pretrained_vocab' : './pretrained_models/RoBERTa/vocab.txt',
    'is_vector' : False,
    'vector' : './vector/bert_vectors_768.txt',
    'embedding_size' : 300,   # embedding dimension     预训练模型：hidden 768   word2voc：300
    'bi_lstm_hidden'  : 300,
    'num_layers' : 1,
    'pad_index': 1,
    'epoch' : 100,
    'batch_size' : 16,
    'chanel_num' : 1,
    'filter_num' : 100,
    'learning_rate' : 2e-4,
    'nhid' : 200, # the dimension of the feedforward network model in nn.TransformerEncoder
    'nlayers' : 2,    # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    'nhead' : 2,  # the number of heads in the multiheadattention models
    'dropout' : 0.2,  # the dropout value
    'key_dim': 64,
    'val_dim': 64,
    'attn_dropout': 0.2,
    'num_heads': 3,
}

class Config():
    def __init__(self, **kwargs):
        super(Config, self).__init__()
        for name, value in default_config.items():
            setattr(self, name, value)
    def add_config(self, config_list):
        for name, value in config_list:
            setattr(self, name, value)

config = Config()