
from tqdm import tqdm
import pandas as pd
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, Trainer, TrainingArguments, BertPreTrainedModel, BertModel, BertTokenizer
import torch
import torch.nn as nn
from dont_patronize_me import DontPatronizeMe
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset, DataLoader

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
      texts.append(b['text'])
      labels.append(b['label'])
    
    encodings = self.tokenizer(
      str(texts), 
      return_tensors = 'pt',
      add_special_tokens = True,
      padding = True,
      truncation = True
      )
    
    encodings['labels'] = torch.tensor(labels)
    return encodings

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):

    item = {'text': self.texts[idx],
            'label': self.labels[idx]}

    return item


class SentimentClassifier(BertPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    # Roberta Model
    self.transformer = BertModel(config)

    # Dropout and Linear Layer
    self.projection_a = nn.Sequential(torch.nn.Dropout(0.2), 
                                      torch.nn.Linear(config.hidden_size, 2))

    self.activation = nn.Sigmoid()
    self.init_weights()

  def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

    output = self.transformer(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    logits = self.projection_a(output[1])
    out = self.activation(logits)

    return out

class Trainer_Sentiment_Classification(Trainer):
  def compute_loss(self, model, inputs):
    
    # get predictions
    outputs = model(**inputs)

    # calculate loss
    loss_fn = nn.CrossEntropyLoss()
    label = inputs['labels']
    loss = loss_fn(outputs.view(-1,2), label.view(-1))

    return loss

def predict_condescending(input,tokenizer,model):
  model.eval()
  encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True)

  output = model(**encodings)
  preds = torch.max(output,1)

  return {'prediction': preds[1], 'confidence': preds[0]}

def evaluate(model, tokenizer, data_loader):

  total_count = 0
  correct_count = 0 

  preds = []
  tot_labels = []

  with torch.no_grad():
    for data in tqdm(data_loader): 
      labels = data['label']
      text = data['text']

      pred = predict_condescending(text, tokenizer, model)
      preds.append(pred['prediction'].tolist())
      tot_labels.append(labels.tolist())

  # with the saved predictions and labels we can compute accuracy, precision, recall and f1-score
  report = classification_report(tot_labels, preds)

  return report

if __name__ == "__main__":

  # Model
  PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
  model = SentimentClassifier.from_pretrained(PRE_TRAINED_MODEL_NAME)

  # Read the data file
  df_train = pd.read_csv('datasets/df_upsample_simple_dup.csv', index_col=0)
  df_test = pd.read_csv('datasets/df_test.csv', index_col=0)
  trainset = reader(df_train)
  testset = reader(df_test)

  # BertTokenizer
  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

  # Get the dataset
  train_dataset = OlidDataset(tokenizer, trainset)
  test_dataset = OlidDataset(tokenizer, testset)

  # Train
  training_args = TrainingArguments(
    output_dir='./experiment/hate_speech',
    learning_rate = 0.0001,
    logging_steps= 100,
    per_device_train_batch_size=16,
    num_train_epochs = 1,
  )

  trainer = Trainer_Sentiment_Classification(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    data_collator = train_dataset.collate_fn
  )
  trainer.train()
  trainer.save_model('./models/bert_finetuned/')

  # Evaluate
  test_loader = DataLoader(test_dataset)
  report = evaluate(model, tokenizer, test_loader)
  print(report)





