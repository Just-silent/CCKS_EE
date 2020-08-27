# CCKS_EE

## environment
```
python      3.6
torch       1.5.0
matplotlib  3.1.1
numpy       1.16.1
```

## 结构
```
base                ：attention实现
data                ：数据及预处理
pretrained_models   ：BERT与训练模型
result              ：保存的结果、图像和对应的 p r f，以及结果的处理
save_model          ：保存的模型
vocab               ：提供的实体词典
config.py           ：配置信息
loss.py             ：dice_loss的实现
model.py            ：模型
module.py           ：模型训练预测等相关函数
result_analysis.py  ：模型结果分析
run.py              ：模型训练入口
test.py             ：测试
tool.py             ：各种工具
```

## data 各个文件及处理分析函数
```
analysis.py                         # 多种分析函数
precess.py                          # 预处理
task2_train_reformat.xlsx           # 原始数据
task2_train_reformat_cleaned.xlsx   # 清洗后的数据
sub_train.xlsx                      # 8-2随机切分后数据
sub_cut_train.xlsx                  # 病例文档切分成子句后的子句文档
```

## run.py 入口文件
```
config.experiment_name = 'test_init_model'      # 实验名称
config.model_name = 'BiLSTM_CRF'                # 模型名称
config.is_vector = False                        # 是否使用bert词向量
config.is_hidden_tag = False                    # 是否增加 子句hidden-> 是否有待抽取属性 的约束
注：此处也可更改或添加其他config.py文件中未涉及到的属性
```

## module.py 训练预测等相关函数
```
train()                 # 训练函数
eval()                  # 评测函数
predict_test()          # 预测test文件
predict_sentence()      # 预测具体医疗病例
```

## 2020.8.22
```
1.embedding
2.data argument
3.ruler
4.dice loss
5.hidden tag & size replace & bioes & CNN
6.autoencoder
7.data clean 
note:find mistakes
```