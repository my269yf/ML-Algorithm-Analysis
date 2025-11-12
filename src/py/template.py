# -*- coding: utf-8 -*-
import pandas as pd

# 1. 获取数据
all_pd_data = pd.read_excel("./data/gastric.xlsx", engine="openpyxl")
print(all_pd_data)

# 2. 加载停用词
with open("./data/stop_words.txt", 'r', encoding="utf-8") as f:
    stop_words = list(l.strip() for l in f.readlines())
stop_words.extend(['\n', '（', '）', ' '])
print(stop_words)

# 3. 对中文文本进行分词
import jieba as jb
all_pd_data['Cut_Text'] = all_pd_data['Text'].apply(
    lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stop_words]))
print(all_pd_data)

# 4. 划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(
    all_pd_data['Cut_Text'], 
    all_pd_data['Label'], 
    test_size=0.2, 
    stratify=all_pd_data['Label'])