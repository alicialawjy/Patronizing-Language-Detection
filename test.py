from tqdm import tqdm
import pandas as pd
from transformers import BertModel, BertTokenizer, RobertaConfig, RobertaModel
import torch
import torch.nn as nn
from dont_patronize_me import DontPatronizeMe
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from training import SentimentClassifier

RANDOM_SEED = 42

dpm = DontPatronizeMe('.', '.')
dpm = DontPatronizeMe('.', 'dontpatronizeme_pcl.tsv')

dpm.load_task1()

dpm2 = DontPatronizeMe('.', '.')
dpm2 = DontPatronizeMe('.', 'dontpatronizeme_categories.tsv')

dpm2.load_task2()

df = dpm.train_task1_df
df_cate = dpm2.train_task2_df

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

df_train, df_test = train_test_split(
  df,
  test_size=0.3,
  random_state = RANDOM_SEED
)

df_test, df_val = train_test_split(
  df_test,
  test_size=0.5,
  random_state = RANDOM_SEED
)

df_train.shape, df_val.shape, df_test.shape

model = SentimentClassifier(2)
# print(model)

model.load_state_dict(torch.load("finetuned_roberta_model.pth"))
model.eval()

print(model)
