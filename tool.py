# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/4/26 10:25
import os
import json
import torch
import logging
import random
import numpy as np
import pandas as pd
from config import config, device
from transformers import *
from random import shuffle
import matplotlib.pyplot as plt
from torchtext.vocab import Vectors
from openpyxl import load_workbook, Workbook
from torchtext.data import Field, Dataset, Example, BucketIterator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def x_tokenizer(sentence):
    return [word for word in sentence]

def y_tokenizer(tag: str):
    return [tag]


TEXT = Field(sequential=True, use_vocab=True, tokenize=x_tokenizer, include_lengths=True)
TAG = Field(sequential=True, tokenize=y_tokenizer, use_vocab=True, is_target=True, pad_token=None)
Hidden_TAG = Field(sequential=True, tokenize=y_tokenizer, use_vocab=True, is_target=True, pad_token=None)
# Fields = [('text', TEXT), ('tag', TAG)]
Fields = [('text', TEXT), ('tag', TAG), ('hidden_tag', Hidden_TAG)]


def get_tag(sentence, origin_places, sizes, transfered_places):
    len_sentence = len(sentence)
    tag = ['O' for i in range(len_sentence)]
    tag_kinds = ['origin_place', 'size', 'transfered_place']
    for i, columns in enumerate([origin_places, sizes, transfered_places]):
        if columns is not None:
            for column in columns.split(','):
                # 如果能够直接找到
                start = sentence.find(column)
                end = start + len(column) - 1
                if start==-1:
                    print('找不到')
                # 不能直接找到
                # size_chars = ['C', 'M', 'c', 'm', '*', '×', '.', ' ', 'X']
                # if start == -1:
                #     # 如果是一维数据（eg：35cm，而不是35cm*35cm）
                #     if not column.__contains__('*') and not column.__contains__('×'):
                #         num = ''
                #         for x in column:
                #             if x.isdigit():
                #                 num+=x
                #         start = sentence.find(num)
                #         end = start
                #         while end+1<len(sentence):
                #             if sentence[end+1] in size_chars or sentence[end+1].isdigit():
                #                 end+=1
                #             else:
                #                 break
                #     # 如果是二维数据（eg：不是35cm，而是35cm*35cm）
                #     elif column.__contains__('*') or column.__contains__('×'):
                #         if column.__contains__('*'):
                #             sub1, sub2 = column.split('*')
                #         else:
                #             sub1, sub2 = column.split('×')
                #         num1 = ''
                #         for x in sub1:
                #             if x.isdigit() or x == '.':
                #                 num1 += x
                #         num2 = ''
                #         for x in sub2:
                #             if x.isdigit() or x == '.':
                #                 num2 += x
                #         start = sentence.find(num1)
                #         start2 = sentence.find(num2)
                #         if start2 - start >10:
                #             print('距离过大，寻找错误')
                #         else:
                #             end = start
                #             while True:
                #                 if sentence[end+1] in size_chars or sentence[end+1].isdigit():
                #                     end += 1
                #                 else:
                #                     break
                tag[start] = 'B_{}'.format(tag_kinds[i])
                start += 1
                while start<=end:
                    tag[start] = 'I_{}'.format(tag_kinds[i])
                    start+=1
    return tag

def get_all_tag(sentence, origin_places, sizes, transfered_places):
    len_sentence = len(sentence)
    tag = ['O' for i in range(len_sentence)]
    tag_kinds = ['origin_place', 'size', 'transfered_place']
    for i, columns in enumerate([origin_places, sizes, transfered_places]):
        if columns is not None:
            if i==0 or i==2:
                for column in columns.split(','):
                    starts = tool.find_all_index(sentence, column)
                    ends = [start + len(column) -1 for start in starts]
                    for j in range(len(starts)):
                        x = starts[j]
                        tag[x] = 'B_{}'.format(tag_kinds[i])
                        x += 1
                        while x <= ends[j]:
                            tag[x] = 'I_{}'.format(tag_kinds[i])
                            x += 1
            else:
                for column in columns.split(','):
                    start = 0
                    end = 0
                    start = sentence.find(column)
                    end = start + len(column) -1
                    if start==-1:
                        print('未找到')
                    # size_chars = ['C', 'M', 'c', 'm', '*', '×', '.', ' ', 'X']
                    # flag = False
                    # if not column.__contains__('*') and not column.__contains__('×'):
                    #     num = ''
                    #     for x in column:
                    #         if x.isdigit():
                    #             num += x
                    #     starts = tool.find_all_index(sentence,num)
                    #     if len(starts)>1:
                    #         for start_i in starts:
                    #             if start_i+6<len(sentence):
                    #                 if 'm' in sentence[start_i:start_i+6] or 'M'in sentence[start_i:start_i+6]:
                    #                     flag = True
                    #                     start = start_i
                    #                     end = start
                    #                     while True:
                    #                         if sentence[end+1] in size_chars or sentence[end+1].isdigit():
                    #                             end += 1
                    #                         else:
                    #                             break
                    #                     break
                    #     elif len(starts)==1:
                    #         start = starts[0]
                    #         end = start
                    #         if 'm' in sentence[start:start + 6] or 'M' in sentence[start:start + 6]:
                    #             flag = True
                    #             while True:
                    #                 if sentence[end + 1] in size_chars or sentence[end + 1].isdigit():
                    #                     end += 1
                    #                 else:
                    #                     break
                    # elif column.__contains__('*') or column.__contains__('×'):
                    #     if column.__contains__('*'):
                    #         sub1, sub2 = column.split('*')
                    #     else:
                    #         sub1, sub2 = column.split('×')
                    #     num1 = ''
                    #     for x in sub1:
                    #         if x.isdigit() or x == '.':
                    #             num1 += x
                    #     num2 = ''
                    #     for x in sub2:
                    #         if x.isdigit() or x == '.':
                    #             num2 += x
                    #     start = sentence.find(num1)
                    #     start2 = sentence.find(num2)
                    #     if start2 - start > 10 or start==-1 or start2==-1:
                    #         pass
                    #         # print('距离过大，寻找错误')
                    #     else:
                    #         flag = True
                    #         end = start
                    #         while end+1 <len(sentence):
                    #             if sentence[end + 1] in size_chars or sentence[end + 1].isdigit():
                    #                 end += 1
                    #             else:
                    #                 break
                    # if flag:
                    tag[start] = 'B_{}'.format(tag_kinds[i])
                    start += 1
                    while start <= end:
                        tag[start] = 'I_{}'.format(tag_kinds[i])
                        start += 1
    return tag

class EEDataset(Dataset):
    def __init__(self, path, fields, encoding="utf-8", **kwargs):
        examples = []
        wb = load_workbook(filename=path)
        ws = wb['sheet1']
        max_row = ws.max_row
        for line_num in range(max_row-1):
            line_num = line_num+2
            sentence, origin_places ,sizes ,transfered_places = ws.cell(line_num, 1).value, ws.cell(line_num, 2).value, ws.cell(line_num, 3).value, ws.cell(line_num, 4).value
            hidden_tag = [0]
            if not (origin_places is None and sizes is None and transfered_places is None):
                hidden_tag = [1]
            if sentence is not None:
                tag_list = get_all_tag(sentence, origin_places, sizes, transfered_places)
                sentence_list = [x for x in sentence]
                # examples.append(Example.fromlist((sentence_list, tag_list), fields))
                examples.append(Example.fromlist((sentence_list, tag_list, hidden_tag), fields))
        super(EEDataset, self).__init__(examples, fields, **kwargs)

class Tool():
    def load_data(self, path: str, fields=Fields):
        dataset = EEDataset(path, fields=fields)
        return dataset

    def get_text_vocab(self, *dataset):
        if config.is_vector is False:
            TEXT.build_vocab(*dataset)
        else:
            vec = Vectors(name=config.vector)
            TEXT.build_vocab(*dataset,
                 max_size=3000,
                 min_freq=1,
                 vectors=vec,  #vects替换为None则不使用词向量
                 unk_init = torch.Tensor.normal_)
        return TEXT.vocab

    def get_tag_vocab(self, *dataset):
        TAG.build_vocab(*dataset)
        return TAG.vocab

    def get_hidden_tag_vocab(self, *dataset):
        Hidden_TAG.build_vocab(*dataset)
        return Hidden_TAG.vocab

    def get_iterator(self, dataset: Dataset, batch_size=1,
                     sort_key=lambda x: len(x.text), sort_within_batch=True):
        iterator = BucketIterator(dataset, batch_size=batch_size, sort_key=sort_key,
                              sort_within_batch=sort_within_batch, device=device)
        return iterator

    def _evaluate(self, tag_true, tag_pred):
        """
        先对true进行还原成 [{}] 再对pred进行还原成 [{}]
        :param tag_true: list[]
        :param tag_pred: list[]
        :return:
        """
        true_list = self._build_list_dict(_len=len(tag_true), _list=tag_true)
        pred_list = self._build_list_dict(_len=len(tag_pred), _list=tag_pred)
        entities = {'origin_place': {'TP': 0, 'S': 0, 'G': 0},
                    'size': {'TP': 0, 'S': 0, 'G': 0},
                    'transfered_place': {'TP': 0, 'S': 0, 'G': 0}}
        for true in true_list:
            label_type = true['label_type']
            entities[label_type]['G'] += 1
        for pred in pred_list:
            start_pos = pred['start_pos']
            end_pos = pred['end_pos']
            label_type = pred['label_type']
            entities[label_type]['S'] += 1
            for true in true_list:
                if label_type == true['label_type'] and start_pos == true['start_pos'] and end_pos == true['end_pos']:
                    entities[label_type]['TP'] += 1
        return entities

    def _build_list_dict(self, _len, _list):
        build_list = []
        tag_dict = {'origin_place': 'origin_place',
                    'size': 'size',
                    'transfered_place': 'transfered_place'}
        for index, tag in zip(range(_len), _list):
            if tag[0] == 'B':
                label_type = tag[2:]
                start_pos = index
                if index < _len-1:
                    end_pos = index + 1
                    while _list[end_pos][0] == 'I' and _list[end_pos][2:] == label_type and end_pos<_len-1:
                        end_pos += 1
                else:
                    end_pos = index
                build_list.append({'start_pos': start_pos,
                                   'end_pos': end_pos,
                                   'label_type': tag_dict[label_type]})
        return build_list

    def show_1y(self, list_x, list_y, name):
        fig, ax = plt.subplots()
        plt.xlabel('The updating of {}'.format(name))
        plt.ylabel(name)
        plt.plot(list_x, list_y, label=name)

        """set interval for y label"""
        # yticks = range(0, 100, 10)
        # ax.set_yticks(yticks)

        """set min and max value for axes"""
        # ax.set_ylim([0, 100])
        # ax.set_xlim([1, len(list_x)])
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.savefig('./result/picture/{}/{}.jpg'.format(config.experiment_name, name))
        plt.show()

    def write_csv(self, dict):
        # 这里补充相应的配置信息 1.model 2.细节信息 3.epoch
        tag_list = []
        p_list = []
        r_list = []
        f1_list = []
        s_list = []
        for i, name in enumerate(dict):
            tag_list.append(name)
            p_list.append(dict[name]['precision'])
            r_list.append(dict[name]['recall'])
            f1_list.append(dict[name]['f1-score'])
            s_list.append(dict[name]['support'])
        dataframe = pd.DataFrame({'name': tag_list, 'precision': p_list, 'recall': r_list, 'f1': f1_list, 's_persent': s_list})
        # dataframe.to_csv('./result/classification_report/{}/report.csv'.format(config.experiment_name), index=False, sep=str(','))
        dataframe.to_excel('./result/classification_report/{}/report.xlsx'.format(config.experiment_name), sheet_name='sheet1')

    def show_labels_f1_bar_divide(self, report):
        x_name = []
        support = []
        f1 = []
        sum = 0
        triggers_dict = get_labels_proportion(path='./data/train.json', is_O=False)[0]
        for index, key in enumerate(triggers_dict):
            x_name.append(key)
            f1.append(report[key]['f1-score'])
            sum += triggers_dict[key]
        for index, key in enumerate(triggers_dict):
            support.append(round(triggers_dict[key] / sum, 2))
        width = 20
        for i in range(len(x_name) // width):
            start = i * width
            end = i * width + width
            if end >= len(x_name):
                end = len(x_name)
            tool.labels_f1_bar(x_name[start:end], support[start:end], f1[start:end], 'analyze_{}'.format(i))

    def labels_f1_bar(self, x_name, support, f1, pict_name):
    # def labels_f1_bar(self, x_name, support, f1):
        # 这两行代码解决 plt 中文显示的问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        bar_width = 5  # 条形宽度
        index_support = np.arange(len(x_name)) * 10  # support条形图的横坐标
        index_f1 = index_support + bar_width  # f1条形图的横坐标
        # 使用两次 bar 函数画出两组条形图
        rects1 = plt.bar(index_support, height=support, width=bar_width, color='b', label='占比')
        rects2 = plt.bar(index_f1, height=f1, width=bar_width, color='g', label='f1')
        # 显示对应柱状图的数值
        for rect in rects1:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(height) + '%', ha='center', va='bottom')
        for rect in rects2:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(height) + '%', ha='center', va='bottom')
        plt.legend()  # 显示图例
        plt.xticks(index_support + bar_width / 2, x_name, rotation=-90)  # 让横坐标轴刻度显示 x_name，index_support + bar_width/2 为横坐标轴刻度的位置
        plt.ylim(0, 1)
        plt.ylabel('present')  # 纵坐标轴标题
        plt.title('各个标签占比及其f1')  # 图形标题
        plt.savefig('./result/picture/{}/{}.jpg'.format(config.experiment_name, pict_name))
        plt.show()

    def get_weight(self, sto):
        support = []
        sum = 0
        keys = list(sto.keys())
        triggers_dict = get_labels_proportion(path='./data/train.json', is_O=False)[0]
        for index, key in enumerate(triggers_dict):
            # sum += triggers_dict[key]
            if sum < triggers_dict[key]:
                sum = triggers_dict[key]
        for key in keys[2:]:
            if key not in triggers_dict.keys():
                # support.append(10.000)
                support.append(1.000)
            else:
                # support.append(triggers_dict[key])
                support.append(round(1.25-triggers_dict[key]/(2*sum), 3))
        weight = torch.eye(len(support) + 2)
        weight[0][0] = 1
        weight[1][1] = 1
        for i in range(len(support)):
            i = i + 2
            weight[i][i] = support[i - 2]
        return weight

    def find_all_index(self, str1, str2):
        # 子串str2， 查找目标字符串str1
        starts = []
        start = 0
        over = 0
        while True:
            index = str1.find(str2, start, len(str1))
            if index != -1:
                starts.append(index+over)
                if index+len(str2)<=len(str1)-1:
                    str1 = str1[index+len(str2):]
                    over += index+len(str2)
                else:
                    break
            else:
                break
        return starts

tool = Tool()

