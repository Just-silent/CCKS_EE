# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/7/23 11:51

from module import EE
from config import config

def one_test():
    config.experiment_name = 'BiLSTM_CRF_DAE40'  # 实验名称
    config.model_name = 'BiLSTM_CRF_DAE'  # 模型名称
    config.is_vector = False  # 是否使用bert词向量
    config.is_hidden_tag = False  # 是否增加 子句hidden-> 是否有待抽取属性 的约束
    config.weight= 40

    ee = EE(config)
    ee.train()
    ee.predict_test()
    # ee.predict_sentence()
    # ee.test_format_result()

def many_test():
    test_dict = {
        'experiment_name' : ['BiLSTM_CRF_hidden_tag0.1', 'BiLSTM_CRF_hidden_tag0.3', 'BiLSTM_CRF_hidden_tag0.5', 'BiLSTM_CRF_hidden_tag0.7', 'BiLSTM_CRF_hidden_tag0.9'],
        'model_name' : ['BiLSTM_CRF_hidden_tag', 'BiLSTM_CRF_hidden_tag', 'BiLSTM_CRF_hidden_tag', 'BiLSTM_CRF_hidden_tag', 'BiLSTM_CRF_hidden_tag'],
        'is_vector' : [False, False, False, False, False],
        'is_hidden_tag' : [True, True, True, True, True],
        'weight' : [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    for i in range(len(test_dict['experiment_name'])):
        config.experiment_name = test_dict['experiment_name'][i] # 实验名称
        config.model_name = test_dict['model_name'][i] # 模型名称
        config.is_vector = test_dict['is_vector'][i] # 是否使用bert词向量
        config.is_hidden_tag = test_dict['is_hidden_tag'][i] # 是否增加 子句hidden-> 是否有待抽取属性 的约束
        config.weight = test_dict['weight'][i]

        ee = EE(config)
        ee.train()
        ee.predict_test()
        # ee.predict_sentence()
        # ee.test_format_result()

if __name__ == '__main__':
    many_test()