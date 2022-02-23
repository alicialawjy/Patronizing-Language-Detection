# -*- coding: utf-8 -*-
"""Training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MD7qB_hbVgR43nbACRhL4wfqqEv5pJj3
"""
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

"""### Sample text to visualise tokenisation"""

sample_txt = 'Hello! I love that you are so poor.'
tokens_sample = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens_sample)

"""Then, we do embedding on the tokens. """

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

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

"""### Dataloader"""

class MyDataset(Dataset):

  def __init__(self,dataframe, tokenizer):
    self.x=dataframe.loc[:,['text']]
    self.y=dataframe.loc[:,['label']]

    self.x = np.array(self.x)
    self.y = np.array(self.y)

    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.y)

  def __getitem__(self,idx):
    input = str(self.x[idx])
    labels = self.y[idx]

    encoding = tokenizer.encode_plus(
      input,
      max_length = 200,
      truncation = True, # truncate examples to max length
      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
      return_token_type_ids=False,
      padding = "max_length",
      return_attention_mask=True,
      return_tensors='pt')  # Return PyTorch tensor

    return {
      'text': input,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(labels, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, batch_size):
  ds = MyDataset(
    dataframe = df,
    tokenizer=tokenizer,
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=2
  )

BATCH_SIZE = 32
train_data_loader = create_data_loader(df_train, tokenizer, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, BATCH_SIZE)

"""# Classification"""

GPU = True
if GPU:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
  device = torch.device("cpu")
print(f"Using {device}")

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    configuration = RobertaConfig()
    self.transformer = RobertaModel(configuration)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(768, n_classes)

  def forward(self, input_ids, attention_mask):
    output = self.transformer(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    output = self.drop(output[1])
    return self.out(output)

"""## Training

"""

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  f1_scores = []

  for d in data_loader:
    optimizer.zero_grad()

    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    preds = preds.squeeze()
    # print(f'outputs: {outputs.shape}')
    # print(f'preds: {preds.shape}')
    targets = targets.squeeze()
    # print(f'target: {targets}')
    # print(f'target shape: {targets.shape}')
    loss = loss_fn(outputs, targets)
    # loss.requires_grad = True
    correct_predictions += torch.sum(preds == targets)
    # print(correct_predictions)
    losses.append(loss.item())
    print(loss.item())
    print(f'accuracy per batch = {(torch.sum(preds == targets))/32}')

    target_detach = targets.cpu().detach().numpy()
    preds_detach = preds.cpu().detach().numpy()
    f1_scores.append(f1_score(target_detach.astype(int), preds_detach.astype(int)))

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

  return correct_predictions.double() / n_examples, np.mean(losses), np.mean(f1_scores)

def evaluate(loss_fn):
  losses = []
  correct_predictions = 0
  f1_scores = []
  with torch.no_grad():
      for test_data in test_data_loader:
        input_ids = test_data["input_ids"].to(device)
        attention_mask = test_data["attention_mask"].to(device)
        targets = test_data["targets"].to(device)

        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        preds = preds.squeeze()
        targets = targets.squeeze()
        loss = loss_fn(outputs, targets)
        loss.requires_grad = True
        losses.append(loss.item())
        target_detach = targets.cpu().detach().numpy()
        preds_detach = preds.cpu().detach().numpy()
        f1_scores.append(f1_score(target_detach.astype(int), preds_detach.astype(int)))

        correct_predictions += torch.sum(preds == targets)
  return correct_predictions.double() / len(df_test), np.mean(losses), np.mean(f1_scores)

if __name__ == "__main__":
  EPOCHS = 3

  model = SentimentClassifier(n_classes=2).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
  loss_fn = nn.CrossEntropyLoss().to(device)
  PATH = "finetuned_roberta_model.pth"


  # Main training loop
  train_accuracies = []
  train_losses = []
  train_f1 = []

  test_accuracies = []
  test_losses = []
  test_f1 = []

  for epoch in tqdm(range(EPOCHS)):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss, train_f1 = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      device,
      n_examples = len(df_train)
    )
    print(f'Epoch{epoch}, Train loss {train_loss},  Train accuracy {train_acc}, Train F1 {train_f1}')

    test_acc, test_loss, test_f1 = evaluate(loss_fn)
    print(f'Epoch{epoch}, Test loss {test_loss},  Test accuracy {test_acc}, Train F1 {test_f1}')

  torch.save(model.state_dict(), PATH)

  print(f'Final Train F1 {train_f1}')
  print(f'Final Test F1 {test_f1}')

