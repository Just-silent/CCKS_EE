# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/7/23 11:51

from module import EE
from config import config

def one_test():
    config.experiment_name = 'TransformerEncoderModel_bio'  # 实验名称
    config.model_name = 'TransformerEncoderModel'  # 模型名称
    config.is_vector = False  # 是否使用bert词向量
    config.is_hidden_tag = False  # 是否增加 子句hidden-> 是否有待抽取属性 的约束

    ee = EE(config)
    ee.train()
    ee.predict_test()
    # ee.predict_sentence()
    # ee.test_format_result()

def many_test():
    test_dict = {
        'experiment_name' : ['TransformerEncoderModel_DAE0.1','TransformerEncoderModel_DAE0.3', 'TransformerEncoderModel_DAE0.5','TransformerEncoderModel_DAE0.7', 'TransformerEncoderModel_DAE0.9'],
        'model_name' : ['TransformerEncoderModel_DAE', 'TransformerEncoderModel_DAE', 'TransformerEncoderModel_DAE', 'TransformerEncoderModel_DAE', 'TransformerEncoderModel_DAE'],
        'is_vector' : [False, False, False, False, False],
        'is_hidden_tag' : [False, False, False, False, False],
        'weight' : [0.1,0.3,0.5,0.7,0.9]
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
    # many_test()
    one_test()