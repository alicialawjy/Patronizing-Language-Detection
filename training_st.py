
from unittest.util import _MAX_LENGTH
from tqdm import tqdm
import pandas as pd
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments, BertPreTrainedModel, BertModel, BertTokenizer
import torch
import torch.nn as nn
from dont_patronize_me import DontPatronizeMe
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from simpletransformers.classification import ClassificationModel, ClassificationArgs, MultiLabelClassificationModel, MultiLabelClassificationArgs
from urllib import request
import pandas as pd
import logging
import torch
from collections import Counter
from ast import literal_eval  
from sklearn.metrics import f1_score

def reader(df):
  texts = df['text'].values.tolist()
  labels = df['label'].values.tolist()

  return {'texts': texts, 'labels': labels}


class OlidDataset(Dataset):
  def __init__(self, tokenizer, input_set):
    self.texts = input_set['texts']
    self.labels = input_set['labels']
    self.tokenizer = tokenizer

  def collate_fn(self, batch):
    texts = []
    labels = []

    for b in batch:
      texts.append(str(b['text']))
      labels.append(b['label'])

    encodings = self.tokenizer(
      texts,
      return_tensors = 'pt',
      add_special_tokens = True,
      padding = True,
      truncation = True,
      max_length= 128
      )

    encodings['labels'] = torch.tensor(labels)
    return encodings

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    item = {'text': self.texts[idx], 'label': self.labels[idx]}

    return item


    # logits = self.projection_a(output[1]) # take pooler output layer
    # return logits


def hyperparam_tuning():
  optimizer = ['AdamW']
  learning_rate = [9e-04, 2e-05, 4e-05]
  epochs = [1,2,3]
  
  best_model = None
  f1_score_list  = []
  current_model_no = 0

  param_dict = {}
  curr_best_f1 = 0
  for epoch in epochs:
    for optim in optimizer:
      for lr in learning_rate:
        print("Optim: " + str(optim))
        print("LR: " + str(lr))
        model_args = ClassificationArgs(num_train_epochs=epoch, 
                                          no_save=True, 
                                          no_cache=True, 
                                          overwrite_output_dir=True,
                                          learning_rate=lr,
                                          optimizer=optim)

        model = ClassificationModel("roberta", 
                                      'roberta-base', 
                                      args = model_args, 
                                      num_labels=2, 
                                      use_cuda=cuda_available)


        y_true = df_val['label']

        model.train_model(df_train[['text', 'label']])
        y_pred, _ = model.predict(df_val.text.tolist())
        print(classification_report(y_true, y_pred))
        f1 = f1_score(y_true, y_pred)


        print("F1: " + str(f1))
        param_dict[current_model_no] = [optim, lr, epoch, f1]


        if f1 >= curr_best_f1:
          curr_best_f1 = f1
          best_model = model
  
  try:
    torch.save(best_model, "best_model.pt")

  except:
    pass

  try:
    df = pd.DataFrame.from_dict(param_dict)
    print(df)
    df.to_csv("df_param.csv", orient='index')
  except:
    print(param_dict)

  return best_model
        






if __name__ == "__main__":
  # Fix Device
  GPU = True
  if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device("cpu")
  print(f"Using {device}")

  cuda_available = torch.cuda.is_available()

  # Model
  # PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
  #configuration = RobertaConfig(vocab_size = 50265)
  #model = SentimentClassifier(configuration).to(device)
  #print(model)
  

  # model = SentimentClassifier.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)

  # Read the data file
  

  df_train = pd.read_csv('datasets/balanced_data/df_downsample.csv', index_col=0)
  df_val = pd.read_csv('datasets/df_val.csv', index_col=0)
  df_test = pd.read_csv('datasets/df_test.csv', index_col=0)
  trainset = reader(df_train)
  testset = reader(df_test)


  model = hyperparam_tuning()
  y_pred, _ = model.predict(df_test.text.tolist())
  print(classification_report(df_test['label'],   y_pred))
  

