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

from utils import shift_tokens_right, encode_sentences, noise_sentence
if __name__ == "__main__":
    # Create the hparams dictionary to pass in the model
    # I realise that this isn't really how this is meant to be used, but having this here reminds me that I can edit it when I need
    hparams = argparse.Namespace()

    hparams.freeze_encoder = True
    hparams.freeze_embeds = True
    hparams.eval_beams = 4


    # Load the model
    from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', add_prefix_space=True, use_cache=False)

    bart_model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large", use_cache=False)
        
    # Load the data into the model for training
    summary_data = SummaryDataModule(tokenizer, 'data.csv',
                                     batch_size = 1, num_examples = 100000)

    # Load the model from a pre-saved checkpoint; alternatively use the code below to start training from scratch
    #model = LitModel.load_from_checkpoint("epoch=5.ckpt", learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)

    model = LitModel(learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)



    checkpoint = ModelCheckpoint('./checkpoints/')
    trainer = pl.Trainer(gpus = 1,
                         max_epochs = 20,
                         min_epochs = 20,
                         auto_lr_find = True,
                         checkpoint_callback = checkpoint,
                         progress_bar_refresh_rate = 50,
                         accumulate_grad_batches=16,precision=16)
                         
                         
    # Fit the instantiated model to the data
    trainer.fit(model, summary_data)



    # If you want to manually save a checkpoint, this works, although the model should automatically save (progressively better)
    # checkpoints as it moves through the epochs
trainer.save_checkpoint("./checkpoints/checkpoint-final.ckpt")
