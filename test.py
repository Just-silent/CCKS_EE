# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/4/27 11:03
import torch
import numpy as np
import random
from config import config
from tool import tool

# 读取xlsx文件
# from openpyxl import load_workbook
# # 打开一个workbook
# wb = load_workbook(filename="./data/task2_train_reformat.xlsx")
# # 获取特定的worksheet
# ws = wb.get_sheet_by_name('sheet1')
# #获取表格所有行和列，两者都是可迭代的
# rows = ws.rows
# columns = ws.columns
# # 获取坐标对应的数据
# print(ws.cell(row=1, column=1).value)

# 写入xlsx文件
# from openpyxl import Workbook
# from openpyxl.utils import get_column_letter
# # 在内存中创建一个workbook对象，而且会至少创建一个 worksheet
# wb = Workbook()
# # 获取当前活跃的worksheet,默认就是第一个worksheet
# ws = wb.active
# # 从第2行开始，写入9行10列数据，值为对应的列序号A、B、C、D...
# for row in range(2, 11):
#     for col in range(1, 11):
#         ws.cell(row=row, column=col).value = col
# # 可以使用append插入一行数据
# ws.append(["我", "你", "她"])
# # 保存
# wb.save(filename="./data/test.xlsx")

# not or and
# a = '2.0CM×*3.0CM'
# if not a.__contains__('×') and not a.__contains__('*'):
#     print('既不也不')
# else:
#     print('包含')
# if a.__contains__('×') or a.__contains__('*'):
#     print('或或')

# 测试 for (a,b,c) in (aa,bb,cc):
# aa = [1.1, 1.2, 1.3]
# bb = [2.1, 2.2, 2.3]
# cc = [3.1, 3.2, 3.3]
# for a,b,c in aa,bb,cc:
#     print(a,b,c)

#
# import pandas as pd
# from numpy import *
# import matplotlib.pyplot as plt
# ts = pd.Series(random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
# ts = ts.cumsum()
# ts.plot()
# plt.show()

# 测试train切分数据
# tool.seg_train()

# 测试list的查找下标用法
# list1 = [1]

# test for loading bert
# from transformers import *
# tokenizer = BertTokenizer.from_pretrained(config.pretrained_vocab)
# config_bert = BertConfig.from_pretrained(config.pretrained_config)
# model = BertModel.from_pretrained(config.pretrained_model, config=config_bert)
# id = torch.tensor([tokenizer.encode_plus('既不也不')['input_ids']])
# out = model(id)[0]

# 测试def find_all_index(self, str1, str2)
# print(tool.find_all_index('abcabcabcabc','bc'))

# 测试制表符与空格和tab的区别于联系
# import re
# text = '我叫邢朋举，  来自中原工学院。 有着远大的抱负。       渴望为国家付出自己的微薄之力。'
# texts = re.split('。|\t|',text)
# print(texts)

# 测试边界list的截取
# list1 = [1,2]
# x = max(list1)
# print(list1[x:])

# 测试切分句子函数
# tool.split_text('1、支气管炎、肺气肿;2、左肺上叶肿块考虑周围性肺癌;左肺门淋巴结增大,考虑为转移;3、右肺中叶改变,考虑为发育不全;4、右侧肩胛下内侧弹力纤维瘤;5、气管憩室;6、甲状腺右叶低密度灶;胃窦壁增厚,请结合临床。左肺上叶可见类圆形肿块影,大小约2.0CM×3.0CM,CT值约32HU,增强CT扫描:三期CT值分别为43HU、53HU、75HU,可见部分支气管分支闭塞、狭窄;右肺中叶体积减小,见片状高密度影,内可见轻度扩张支气管影;两肺透过度增强,两肺野内见多发囊状透光区;两肺纹理稀疏、紊乱。左肺门淋巴结稍大,直径约1.4CM。纵隔内多发小淋巴结。两胸腔无积液征象。主动脉及冠脉钙化。右侧肩胛下内侧见片状软组织密度影,约为2.2CM×5.1CM。气管憩室。甲状腺右叶密度减低,强化程度低于正常甲状腺组织。胃窦壁增厚。')
# torch.cuda.empty_cache()

# test for str.replace()
# str = 'w s x p j'
# str.replace('w', 'n')
# # print(str)
# print(str.replace('w', 'n'))

# test
l = []
print(','.join(l))
