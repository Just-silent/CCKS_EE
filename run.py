# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/7/23 11:51

from module import EE
from config import config

def ont_test():
    config.experiment_name = 'BiLSTM_CRF'  # 实验名称
    config.model_name = 'BiLSTM_CRF'  # 模型名称
    config.is_vector = False  # 是否使用bert词向量
    config.is_hidden_tag = False  # 是否增加 子句hidden-> 是否有待抽取属性 的约束

    ee = EE(config)
    # ee.train()
    ee.predict_test()
    # ee.predict_sentence()
    # ee.test_format_result()

def many_test():
    test_dict = {
        'experiment_name' : ['test_all_model_1', 'test_all_model_2','test_all_model_3','test_all_model_4','test_all_model_5','test_all_model_6',],
        'model_name' : ['BiLSTM_CRF', 'BiLSTM_CRF_ATT', 'BiLSTM_CRF_DAE', 'BiLSTM_CRF_hidden_tag', 'TransformerEncoderModel', 'TransformerEncoderModel_DAE'],
        'is_vector' : [False, False, False, False, False, False],
        'is_hidden_tag' : [False, False, False, True, False, False]
    }
    for i in range(len(test_dict['experiment_name'])):
        config.experiment_name = test_dict['experiment_name'][i] # 实验名称
        config.model_name = test_dict['model_name'][i] # 模型名称
        config.is_vector = test_dict['is_vector'][i] # 是否使用bert词向量
        config.is_hidden_tag = test_dict['is_hidden_tag'][i] # 是否增加 子句hidden-> 是否有待抽取属性 的约束

        ee = EE(config)
        ee.train()
        ee.predict_test()
        # ee.predict_sentence()
        # ee.test_format_result()

if __name__ == '__main__':
    ont_test()