# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/4/26 10:25

import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

default_config = {
    'experiment_name' : 'test1',                                    # 实验名称
    'model_path': './save_model/{}.pkl',                            # 保存模型位置
    'analysis_path' : './result/data/{}/analysis.xlsx',
    'train_path' : './data/sub_cut_train.xlsx',
    'dev_path' : './data/sub_cut_dev.xlsx',
    'train_dev_path' : './data/task2_train_reformat_cleaned.xlsx',
    'test_path' : './data/task2_no_val_cleaned.xlsx',
    'unformated_val_path' : './result/data/{}/unformated_val.xlsx',  # 模型训练直接预测
    'test_formated_val_path' : './result/data/{}/test_format/formated_val.xlsx',    # 测试format结果是否有提升
    'test_unformated_val_path' : './result/data/{}/test_format/unformated_val.xlsx',
    'model_name' : 'BiLSTM_CRF_hidden_tag',# CNN_CRF、BiLSTM_CRF、BiLSTM_CRF_ATT、BiLSTM_CRF_DAE、BiLSTM_CRF_hidden_tag、TransformerEncoderModel、TransformerEncoderModel_DAE
    'is_hidden_tag' : False,                                        # 是否增加 子句hidden-> 是否有待抽取属性 的约束
    'is_pretrained_model' : False,                                  # 是否使用预训练模型
    'pretrained_config' : './pretrained_models/RoBERTa/config.json',
    'pretrained_model' : './pretrained_models/RoBERTa/pytorch_model.bin',
    'pretrained_vocab' : './pretrained_models/RoBERTa/vocab.txt',
    'is_vector' : False,                                            # 是否使用词向量
    'vector' : './vector/bert_vectors_768.txt',
    'embedding_size' : 300,   # embedding dimension     预训练模型：hidden 1024/786   word2voc：300
    'bi_lstm_hidden'  : 300,
    'num_layers' : 1,
    'pad_index': 1,
    'weight':1,
    'epoch' : 100,
    'batch_size' : 64,
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