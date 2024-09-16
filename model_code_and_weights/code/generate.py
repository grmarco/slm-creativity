# imports
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np

import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import SummaryDataModule 
from trainer import LitModel

import math
import random
import re
import argparse

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig


def generate_lyrics(seed_line, num_lines, model_, noise_percent = 0.25, multiple_lines = False, max_line_history = 3):
  ''' Function that generates lyrics based on previously generated lyrics 
      Args: seed_line - a line to start off the machine
            num_lines - the number of lines to generate
            model_ - the model used to generate the text
            multiple_lines - whether the model generates based on multiple previous lines or just the past line
            max_line_history - the maximum number of previous lines used in the current input
      Returns a list with num_lines of rap lines
  '''
  # Put the model on eval mode
  model_.to(torch.device('cpu'))
  model_.eval()
  lyrics = []
  lyrics.append(seed_line)
  prompt_line_tokens = tokenizer(noise_sentence(seed_line, 0.2), max_length = 256, return_tensors = "pt", truncation = True)
  # Loop through the number of lines generating a new line based on the old

  line = [seed_line]
  for i in range(num_lines):
    # Print out the new line
    print(line[0].strip())
    lyrics.append(line[0])
    line = model.generate_text(prompt_line_tokens, eval_beams = 4, max_len=256)
    # This deals with an artefact in the training data that I had an issue cleaning
    if line[0].find(":") != -1:
      line[0] = re.sub(r'[A-Z]+: ', '', line[0])
    # This allows the model to generate a new line conditioned on more than one line
    if multiple_lines:
      start_line = np.maximum(0, i - max_line_history)
      end_line = i
      prompt_line = ' '.join(lyrics[start_line:end_line]) # Going to end_line is fine because it is non-inclusive
    else:
      prompt_line = lyrics[i]
    prompt_line_tokens = tokenizer(prompt_line, max_length = 200, return_tensors = "pt", truncation = True)

  return lyrics
 
 
def generate_line(text, length):
    prompt_line_tokens = tokenizer(text, max_length=max_length, return_tensors = "pt", truncation = True)
    lines = model.generate_text(prompt_line_tokens, eval_beams = 5, max_len=512)
    best_line = 10e5
    final_line = ""
    for line in lines:
        len_line = len(line.split())
        if abs(len_line-length) < abs(best_line-length):
            
            best_line = len_line
            final_line = line
    return final_line

if __name__ == "__main__":


    # Create the hparams dictionary to pass in the model
    # I realise that this isn't really how this is meant to be used, but having this here reminds me that I can edit it when I need
    hparams = argparse.Namespace()

    hparams.freeze_encoder = True
    hparams.freeze_embeds = True
    hparams.eval_beams = 4
    hparams.max_length=512
    hparams.max_position_embeddings = 512

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True, max_position_embeddings=512)
    bart_model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-base")
    model = LitModel.load_from_checkpoint("epoch=5.ckpt", learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)
     
    max_length = 512
    model.to(torch.device('cpu'))
    model.eval()
    df_test = pd.read_csv('data/test.csv')


    df_test['bart_text'] = df_test.apply(lambda x: generate_line(x['source'], x['len']), axis=1)
    df_test['len_bart'] = df_test['bart_text'].str.split().apply(len)
    df_test.to_csv('test-gen.csv')

