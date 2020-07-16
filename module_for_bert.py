import os
import numpy
import torch
from tqdm import tqdm
from config import config
from tool import tool, logger
from openpyxl import load_workbook
from model import TransformerEncoderModel
from sklearn.metrics import classification_report
from result.predict_eval_process import format_result
import torch.optim as optim
from transformers import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import warnings
warnings.filterwarnings('ignore')

class Paddle_EE():
    def __init__(self):
        self.model = None
        self.word_vocab = None
        self.tag_vocab = None
        self.train_dev_data = None

    def train(self):
        max_f1 = -1
        max_dict = {}
        max_report = {}
        loss_list = []
        f1_list = []
        epoch_list = []
        if not os.path.exists('./result/classification_report/{}'.format(config.experiment_name)):
            os.mkdir('./result/classification_report/{}'.format(config.experiment_name))
            os.mkdir('./result/picture/{}'.format(config.experiment_name))
            os.mkdir('./result/data/{}'.format(config.experiment_name))
            os.mkdir('./result/data/{}/test_format'.format(config.experiment_name))
        logger.info('Loading data ...')
        train_data = tool.load_data(config.train_path)
        dev_data = tool.load_data(config.dev_path)
        self.tag_vocab = tool.get_tag_vocab(train_data, dev_data)
        logger.info('Finished load data')
        logger.info('Building vocab ...')
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_vocab)
        self.tag_vocab = tool.get_tag_vocab(train_data, dev_data)
        logger.info('Finished build vocab')
        model = TransformerEncoderModel(config, len(self.tag_vocab)).to(device)
        self.model = model
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        logger.info('Begining train ...')
        for epoch in range(config.epoch):
            model.train()
            acc_loss = 0
            for index, example in enumerate(tqdm(train_data.examples)):
                optimizer.zero_grad()
                encoded_dict = tokenizer.encode_plus(
                    example.text,  # Sentence to encode.
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=512,  # Pad & truncate all sentences.
                    pad_to_max_length=True,
                    return_attention_mask=True,  # Construct attn. masks.
                    return_tensors='pt',  # Return pytorch tensors.
                )
                text = encoded_dict['input_ids'].transpose(0,1).to(device)
                attention_mask =  encoded_dict['attention_mask'].transpose(0,1).to(device)
                tag = self.get_tag(example)
                text_len = torch.tensor([len(example.text)+2 if len(example.text)<=510 else 512]).to(device)
                loss = (-model.loss(text, attention_mask, text_len, tag)) / tag.size(1)
                acc_loss = loss.view(-1).cpu().data.tolist()[0]
                acc_loss += loss.item()
                loss.backward()
                optimizer.step()
            f1, report_dict, entity_prf_dict = self.eval(dev_data, tokenizer)
            entity_prf_dict['labels_weighted_avg'] = report_dict['weighted avg']
            loss_list.append(acc_loss)
            f1_list.append(f1)
            epoch_list.append(epoch+1)
            logger.info('epoch:{}   loss:{}   weighted avg:{}'.format(epoch, acc_loss, report_dict['weighted avg']))
            if f1 > max_f1:
                max_f1 = f1
                max_dict = entity_prf_dict['average']
                max_report = entity_prf_dict
                torch.save(model.state_dict(), './save_model/{}.pkl'.format(config.experiment_name))
        logger.info('Finished train')
        logger.info('Max_f1 weighted avg : {}'.format(max_dict))
        tool.write_csv(max_report)
        tool.show_1y(epoch_list, loss_list, 'loss')
        tool.show_1y(epoch_list, f1_list, 'f1')
        # 稍后处理
        # tool.show_labels_f1_bar_divide(max_report)

    def get_tag(self, example):
        if len(example.tag)<510:
            list1 = [0]
            for index, tag in enumerate(example.tag):
                if index<len(example.tag):
                    list1.append(self.tag_vocab.stoi[tag])
            while index+1>=len(example.tag) and index+1<510:
                list1.append(0)
                index+=1
            list1.append(0)
            return torch.tensor([list1]).transpose(0,1).to(device)
        else:
            list1 = [0]
            for index, tag in enumerate(example.tag):
                if index <= 509:
                    list1.append(self.tag_vocab.stoi[tag])
            list1.append(0)
            return torch.tensor([list1]).transpose(0,1).to(device)

    def eval(self, dev_data, tokenizer):
        self.model.eval()
        tag_pred = []
        tag_true = []
        tag_true_all = []
        tag_pred_all = []
        entity_prf_dict = {}
        entities_total = {'origin_place': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          'size': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          'transfered_place': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0}}
        for index, example in enumerate(tqdm(dev_data.examples)):
            model = self.model
            encoded_dict = tokenizer.encode_plus(
                example.text,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            text = encoded_dict['input_ids'].transpose(0, 1).to(device)
            tag = self.get_tag(example).transpose(0,1)
            attention_mask = encoded_dict['attention_mask'].transpose(0, 1).to(device)
            text_len = torch.tensor([len(example.text)+2 if len(example.text) <= 510 else 512]).to(device)
            result = model(text, attention_mask, text_len)
            for i, result_list in zip(range(text.size(1)), result):
                tag_list = tag[i][1:text_len[i].item()-1]
                result_list = result_list[1:-1]
                assert len(tag_list) == len(result_list), 'tag_list: {} != result_list: {}'.format(len(tag_list),
                                                                                                len(result_list))
                tag_true = [self.tag_vocab.itos[k] for k in tag_list]
                tag_true_all.extend(tag_true)
                tag_pred = [self.tag_vocab.itos[k] for k in result_list]
                tag_pred_all.extend(tag_pred)

                entities = self._evaluate(tag_true=tag_true, tag_pred=tag_pred)
                assert len(entities_total) == len(entities), 'entities_total: {} != entities: {}'.format(
                    len(entities_total), len(entities))
                for entity in entities_total:
                    entities_total[entity]['TP'] += entities[entity]['TP']
                    entities_total[entity]['S'] += entities[entity]['S']
                    entities_total[entity]['G'] += entities[entity]['G']
        TP = 0
        S = 0
        G = 0
        print('\n--------------------------------------------------')
        print('\tp\t\t\tr\t\t\tf1\t\t\tlabel_type')
        for entity in entities_total:
            entities_total[entity]['p'] = entities_total[entity]['TP'] / entities_total[entity]['S'] \
                if entities_total[entity]['S'] != 0 else 0
            entities_total[entity]['r'] = entities_total[entity]['TP'] / entities_total[entity]['G'] \
                if entities_total[entity]['G'] != 0 else 0
            entities_total[entity]['f1'] = 2 * entities_total[entity]['p'] * entities_total[entity]['r'] / \
                                           (entities_total[entity]['p'] + entities_total[entity]['r']) \
                if entities_total[entity]['p'] + entities_total[entity]['r'] != 0 else 0
            print('\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{}'.format(entities_total[entity]['p'], entities_total[entity]['r'],
                                                              entities_total[entity]['f1'], entity))
            entity_dict = {'precision':entities_total[entity]['p'], 'recall':entities_total[entity]['r'], 'f1-score':entities_total[entity]['f1'], 'support':''}
            entity_prf_dict[entity] = entity_dict
            TP += entities_total[entity]['TP']
            S += entities_total[entity]['S']
            G += entities_total[entity]['G']
        p = TP / S if S != 0 else 0
        r = TP / G if G != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        print('\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\taverage'.format(p, r, f1))
        print('--------------------------------------------------')
        entity_prf_dict['average'] = {'precision':p, 'recall':r, 'f1-score':f1, 'support':''}
        labels = []
        for index, label in enumerate(self.tag_vocab.itos):
            labels.append(label)
        labels.remove('O')
        prf_dict = classification_report(tag_true_all, tag_pred_all, labels=labels, output_dict=True)
        return f1, prf_dict, entity_prf_dict

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
                start_pos = index
                if index < _len-1:
                    end_pos = index + 1
                    label_type = tag[2:]
                    while _list[end_pos][0] == 'I' and _list[end_pos][2:] == label_type and end_pos<_len-1:
                        end_pos += 1
                else:
                    end_pos = index
                build_list.append({'start_pos': start_pos,
                                   'end_pos': end_pos,
                                   'label_type': tag_dict[label_type]})
        return build_list
if __name__ == '__main__':
    paddle_ee = Paddle_EE()
    paddle_ee.train()