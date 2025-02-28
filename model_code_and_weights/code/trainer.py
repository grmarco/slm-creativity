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

class LitModel(pl.LightningModule):
  # Instantiate the model
  def __init__(self, learning_rate, tokenizer, model, hparams):
    super().__init__()
    self.tokenizer = tokenizer
    self.model = model
    self.learning_rate = learning_rate
    # self.freeze_encoder = freeze_encoder
    # self.freeze_embeds_ = freeze_embeds
    self.hparams = hparams

    if self.hparams.freeze_encoder:
      freeze_params(self.model.get_encoder())

    if self.hparams.freeze_embeds:
      self.freeze_embeds()
  
  def freeze_embeds(self):
    ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
    freeze_params(self.model.model.shared)
    for d in [self.model.model.encoder, self.model.model.decoder]:
      freeze_params(d.embed_positions)
      freeze_params(d.embed_tokens)

  # Do a forward pass through the model
  def forward(self, input_ids, **kwargs):
    return self.model(input_ids, **kwargs)
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
    return optimizer

  def training_step(self, batch, batch_idx):
    # Load the data into variables
    src_ids, src_mask = batch[0], batch[1]
    tgt_ids = batch[2]
    # Shift the decoder tokens right (but NOT the tgt_ids)
    decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
    lm_logits = outputs[0]
    # Create the loss function
    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    # Calculate the loss on the un-shifted tokens
    loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

    return {'loss':loss}

  def validation_step(self, batch, batch_idx):

    src_ids, src_mask = batch[0], batch[1]
    tgt_ids = batch[2]

    decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
    
    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
    lm_logits = outputs[0]

    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
    print('Val loss',val_loss)
    return {'loss': val_loss}
  
  # Method that generates text using the BartForConditionalGeneration's generate() method
  def generate_text(self, text, eval_beams, early_stopping = True, max_len = 40):
    ''' Function to generate text '''
    generated_ids = self.model.generate(
        text["input_ids"],
        attention_mask=text["attention_mask"],
        use_cache=True,
	num_return_sequences = 5,
        decoder_start_token_id = self.tokenizer.pad_token_id,
        num_beams= eval_beams,
        max_length = max_len,
        early_stopping = early_stopping
    )
    return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]

def freeze_params(model):
  ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
      adapted from finetune.py '''
  for layer in model.parameters():
    layer.requires_grade = False

