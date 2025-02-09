import os, sys, random, re, collections, string
import numpy as np
import torch
import math
import csv
import sklearn.model_selection
import sklearn.metrics
import heapq
import matplotlib
import tqdm
import transformers

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast

## 1.1 Tokenize
########################

boolq_dataset = load_dataset('google/boolq')
emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence', trust_remote_code=True)

# Not sure
# gpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained('distilbert/distilgpt2')
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('distilbert/distilgpt2')
# print(gpt2_tokenizer.tokenize("This is a test of tokenizing from Stony Brook University."))

boolq_train_passages = boolq_dataset['train']['passage']
boolq_train_questions = boolq_dataset['train']['question']
boolq_train_answers = boolq_dataset['train']['answer']
# Space after question?
boolq_train_rows = [f"{p} {q}? {a}" for p, q, a in zip(boolq_train_passages, boolq_train_questions, boolq_train_answers)]
boolq_train_tokens = [gpt2_tokenizer.tokenize(row) for row in boolq_train_rows]

emo_train_texts = emo_dataset['train']['text']
emo_train_tokens = [gpt2_tokenizer.tokenize(row) for row in emo_train_texts]

if __name__ == "__main__":
    print("\nCheckpoint 1.1:")
    print("\nBoolQ")
    print(f"first: {boolq_train_tokens[0]}")
    print(f"last: {boolq_train_tokens[-1]}")

    print("\nEmoBank")
    print(f"first: {emo_train_tokens[0]}")
    print(f"last: {emo_train_tokens[-1]}")


## 1.2 Smoothed Trigram Language Model
########################

class TrigramLM:
    def __init__(self):
        self.unigram_counts = collections.defaultdict(int)
        self.trigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        self.total_tokens = 0
        self.vocab = {"OOV"}

    def train(self, datasets):
        for dataset in datasets.values():
            # print(dataset)
            for line_tokens in dataset:
                line_tokens = ['<s>'] + line_tokens + ['</s>']
                self.total_tokens += len(line_tokens)

                for token in line_tokens:
                    self.unigram_counts[token] += 1
                    self.vocab.add(token)

                for w1, w2, w3 in zip(line_tokens, line_tokens[1:], line_tokens[2:]):
                    self.trigram_counts[w1 + " " + w2][w3] += 1

        # print(self.unigram_counts)
        # print(self.trigram_counts)

    def nextProb(self, history_toks, next_toks):
        all_probs = [-1 for i in range(len(next_toks))]
        V = len(self.vocab)

        # What to do if next tok is an OOV?
        for i, next_tok in enumerate(next_toks):
            # print(next_tok, self.unigram_counts[next_tok])
            unigram_prob = (self.unigram_counts[next_tok] + 1) / (self.total_tokens + V)
            all_probs[i] = unigram_prob
        
        if len(history_toks) >= 2:
            for i, next_tok in enumerate(next_toks):
                current_bigram = " ".join(history_toks[-2:])
                current_bigram_tokens = sum(self.trigram_counts[current_bigram].values())
                trigram_prob = (self.trigram_counts[current_bigram][next_tok] + 1) / (current_bigram_tokens + V)
                all_probs[i] = (all_probs[i] + trigram_prob) / 2
        
        return all_probs

trigramLM = TrigramLM()
trigramLM.train({'boolq': boolq_train_tokens, 'emo': emo_train_tokens})

if __name__ == "__main__":
    history_toks_sample1 = ['is', 'Ġmargin', 'Ġof', 'Ġerror', 'Ġthe', 'Ġsame', 'Ġas', 'Ġconfidence']
    history_toks_sample2 = ['Ġby', 'Ġland', 'Ġor', 'Ġby']
    next_toks_sample1 = ['Ġinterval', 'Ġthe', 'Ġis']
    next_toks_sample2 = ['Ġsea', 'Ġwater','Ġcycle']

    all_probs_sample1 = trigramLM.nextProb(history_toks_sample1, next_toks_sample1)
    all_probs_sample2 = trigramLM.nextProb(history_toks_sample1, next_toks_sample2)

    print("\nCheckpoint 1.2:")
    print(f"\nhistory: {history_toks_sample1}")
    for i, p in enumerate(all_probs_sample1):
        print(f"\tword{i+1}: {p:.5f}")

    print(f"\nhistory: {history_toks_sample2}")
    for i, p in enumerate(all_probs_sample2):
        print(f"\tword{i+1}: {p:.5f}")