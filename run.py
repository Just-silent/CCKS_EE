# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/7/23 11:51

from module import EE
from config import config

def one_test():
    config.experiment_name = 'TransformerEncoderModel_bs_64_result_data_argu'  # 实验名称 TransformerEncoderModel_add_loss_code
    config.model_name = 'TransformerEncoderModel'  # 模型名称  TransformerEncoderModel
    config.is_vector = False  # 是否使用bert词向量
    config.is_hidden_tag = False  # 是否增加 子句hidden-> 是否有待抽取属性 的约束
    config.is_bioes = False
    config.is_dice_loss = False
    config.dice_loss_weight = 0.4
    config.is_dae_loss = False
    config.dae_loss_weight = 1
    config.epoch = 100

    ee = EE(config)
    ee.train()
    ee.predict_test()
    # ee.write_val_true_pred()
    # ee.predict_sentence()
    # ee.test_format_result()

def many_test():
    # data_unclean
    # data_clean
    # bioes
    # bio
    # replace_size
    # CNN+
    # bert
    test_dict = {
        'experiment_name' : ['TransformerEncoderModel_dice_loss_400', 'TransformerEncoderModel_dice_loss_700'],
        'model_name' : ['TransformerEncoderModel', 'TransformerEncoderModel'],
        'is_vector' : [False, False],
        'is_bioes' : [False, False],
        'embedding_size' : [300, 300],
        'weight':[400,700]
    }
    for i in range(len(test_dict['experiment_name'])):
        config.experiment_name = test_dict['experiment_name'][i] # 实验名称
        config.model_name = test_dict['model_name'][i] # 模型名称
        config.is_vector = test_dict['is_vector'][i] # 是否使用bert词向量
        config.is_bioes = test_dict['is_bioes'][i]
        config.embedding_size = test_dict['embedding_size'][i]
        config.is_dice_loss = True
        config.dice_loss_weight = test_dict['weight'][i]
        config.epoch = 100

        ee = EE(config)
        ee.train()
        ee.predict_test()
        # ee.predict_sentence()
        # ee.test_format_result()

if __name__ == '__main__':
    one_test()