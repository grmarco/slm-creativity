# imports
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np

import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import math
import random
import re
import argparse

from utils import shift_tokens_right, encode_sentences, noise_sentence

# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
  def __init__(self, tokenizer, data_file, batch_size, num_examples = 20000):
    super().__init__()
    self.tokenizer = tokenizer
    self.data_file = data_file
    self.batch_size = batch_size
    self.num_examples = num_examples
  
  # Loads and splits the data into training, validation and test sets with a 60/20/20 split
  def prepare_data(self):
    #self.data = pd.read_csv(self.data_file)[:self.num_examples]

    self.train = pd.read_csv('data/train.csv', lineterminator='\n')[:self.num_examples]
    len_train = len(self.train)
    num_test = int(len_train*(self.num_examples/len_train))
    self.validate = pd.read_csv('data/validate.csv', lineterminator='\n')[:num_test]
    self.test = pd.read_csv('data/test.csv', lineterminator='\n')[:num_test]

  # encode the sentences using the tokenizer  
  def setup(self, stage):
    self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'])
    self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'])
    self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'])

  # Load the training, validation and test sets in Pytorch Dataset objects
  def train_dataloader(self):
    dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
    train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size, num_workers=12)
    return train_data

  def val_dataloader(self):
    dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels']) 
    val_data = DataLoader(dataset, batch_size = self.batch_size, num_workers=12)                       
    return val_data

  def test_dataloader(self):
    dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels']) 
    test_data = DataLoader(dataset, batch_size = self.batch_size, num_workers=12)                   
    return test_data


