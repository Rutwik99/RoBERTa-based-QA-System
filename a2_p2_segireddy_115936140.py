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
import matplotlib.pyplot as plt

import tqdm
from datasets import load_dataset
from a2_p1_segireddy_115936140 import trigramLM

## 2.1. Find the zero-shot accuracy of distil-gpt2 on boolQ 
########################
print("\nCheckpoint: 2.1:")

tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")

model = transformers.AutoModelForCausalLM.from_pretrained("distilgpt2")

boolq_dataset = load_dataset('google/boolq')
boolq_validation_passages = boolq_dataset['validation']['passage']
boolq_validation_questions = boolq_dataset['validation']['question']
boolq_validation_answers = boolq_dataset['validation']['answer']
boolq_validation_rows = [f"{p}\n{q}?\n" for p, q in zip(boolq_validation_passages, boolq_validation_questions)]

# https://github.com/christianversloot/machine-learning-articles/blob/main/easy-causal-language-modeling-with-machine-learning-and-huggingface-transformers.md
def base_boolq_predict_probability(text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model(inputs)
    last_predictions = outputs.logits[0, -1]

    all_probs = torch.nn.functional.softmax(last_predictions, dim=-1)
    prob_yes = all_probs[tokenizer.encode('yes')[0]]
    prob_no = all_probs[tokenizer.encode('no')[0]]
    return prob_yes.item(), prob_no.item()


boolq_gpt2_pred_validation_answers = []
for i in range(len(boolq_validation_rows)):
    prob_yes, prob_no = base_boolq_predict_probability(boolq_validation_rows[i])
    boolq_gpt2_pred_validation_answers.append(prob_yes > prob_no)

accuracy_gpt2 = sklearn.metrics.accuracy_score(boolq_validation_answers, boolq_gpt2_pred_validation_answers)
_, _, f1_gpt2, _ = sklearn.metrics.precision_recall_fscore_support(boolq_validation_answers, boolq_gpt2_pred_validation_answers, average='macro')

precision_class_specific_gpt2, recall_class_specific_gpt2, f1_class_specific_gpt2, _ = sklearn.metrics.precision_recall_fscore_support(boolq_validation_answers, boolq_gpt2_pred_validation_answers, labels=[True, False])

print(f"\nOverall: acc: {accuracy_gpt2:.3f}, f1: {f1_gpt2:.3f}")
print(f"Yes: prec: {precision_class_specific_gpt2[0]:.3f}, rec: {recall_class_specific_gpt2[0]:.3f}, f1: {f1_class_specific_gpt2[0]:.3f}")
print(f"No: prec: {precision_class_specific_gpt2[1]:.3f}, rec: {recall_class_specific_gpt2[1]:.3f}, f1: {f1_class_specific_gpt2[1]:.3f}")


## 2.2. Compare to the zero-shot accuracy your trigramLM
########################
print("\nCheckpoint: 2.2:")

boolq_trigram_pred_validation_answers = []
for i in range(len(boolq_validation_rows)):
    prob_yes, prob_no = trigramLM.nextProb(boolq_validation_rows[i], ["yes", "no"])
    # print(trigramLM.nextProb(boolq_validation_rows[i], ["yes", "no"]))
    boolq_trigram_pred_validation_answers.append(prob_yes > prob_no)

# print(boolq_trigram_pred_validation_answers)
accuracy_trigram = sklearn.metrics.accuracy_score(boolq_validation_answers, boolq_trigram_pred_validation_answers)
_, _, f1_trigram, _ = sklearn.metrics.precision_recall_fscore_support(boolq_validation_answers, boolq_trigram_pred_validation_answers, average='macro')

precision_class_specific_trigram, recall_class_specific_trigram, f1_class_specific_trigram, _ = sklearn.metrics.precision_recall_fscore_support(boolq_validation_answers, boolq_trigram_pred_validation_answers, labels=[True, False])

print(f"Overall: acc: {accuracy_trigram:.3f}, f1: {f1_trigram:.3f}")
print(f"Yes: prec: {precision_class_specific_trigram[0]:.3f}, rec: {recall_class_specific_trigram[0]:.3f}, f1: {f1_class_specific_trigram[0]:.3f}")
print(f"No: prec: {precision_class_specific_trigram[1]:.3f}, rec: {recall_class_specific_trigram[1]:.3f}, f1: {f1_class_specific_trigram[1]:.3f}")


## 2.3. Finetune the distil-gpt2 LM on boolQ language
########################
tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2", truncation_side="left")
tokenizer.pad_token = tokenizer.eos_token

finetuned_model = transformers.GPT2LMHeadModel.from_pretrained("distilgpt2")

boolq_dataset = load_dataset('boolq')
boolq_validation_passages = boolq_dataset['validation']['passage']
boolq_validation_questions = boolq_dataset['validation']['question']
boolq_validation_answers = boolq_dataset['validation']['answer']
boolq_validation_rows = [f"{p}\n{q}?\n" for p, q in zip(boolq_validation_passages, boolq_validation_questions)]

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=1024)

train_dataset = boolq_dataset['train'].map(
    lambda e: {'text': f"{e['passage']}\n{e['question']}?\n{'yes' if e['answer'] else 'no'}"},
    remove_columns=['passage', 'question', 'answer']
)
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

optimizer = torch.optim.AdamW(finetuned_model.parameters(), lr=1e-5, weight_decay=0.01, eps=1e-8)

device = torch.device("cuda")
finetuned_model.to(device)

num_epochs = 2
accumulation_steps = 4
loss_values = []
for epoch in range(num_epochs):
    total_loss = 0.0
    finetuned_model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = finetuned_model(**batch, labels=batch['input_ids'])

        loss = outputs.loss
        # loss = loss / accumulation_steps
        loss.backward()
        total_loss += loss.item()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
        #     loss_values.append(total_loss)
        #     total_loss = 0.0
        loss_values.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

plt.plot(loss_values)
plt.title('Training loss over time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('23.png')
plt.show()

## 2.4. Compute the zero-shot accuracy on boolQ language
########################

def finetuned_boolq_predict_probability(text):
    finetuned_model.eval()
    inputs = tokenizer.encode(text, truncation=True, max_length=1024, return_tensors="pt")
    outputs = finetuned_model(inputs.to(device))
    last_predictions = outputs.logits[0, -1]

    all_probs = torch.nn.functional.softmax(last_predictions, dim=-1)
    prob_yes = all_probs[tokenizer.encode('yes')[0]]
    prob_no = all_probs[tokenizer.encode('no')[0]]
    return prob_yes.item(), prob_no.item()

boolq_finetuned_pred_validation_answers = []
for i in range(len(boolq_validation_rows)):
    prob_yes, prob_no = finetuned_boolq_predict_probability(boolq_validation_rows[i])
    boolq_finetuned_pred_validation_answers.append(prob_yes > prob_no)

accuracy_finetuned = sklearn.metrics.accuracy_score(boolq_validation_answers, boolq_finetuned_pred_validation_answers)
_, _, f1_finetuned, _ = sklearn.metrics.precision_recall_fscore_support(boolq_validation_answers, boolq_finetuned_pred_validation_answers, average='macro')

precision_class_specific_finetuned, recall_class_specific_finetuned, f1_class_specific_finetuned, _ = sklearn.metrics.precision_recall_fscore_support(boolq_validation_answers, boolq_finetuned_pred_validation_answers, labels=[True, False])

print("\nCheckpoint: 2.4:")
print(f"Overall: acc: {accuracy_finetuned:.3f}, f1: {f1_finetuned:.3f}")
print(f"Yes: prec: {precision_class_specific_finetuned[0]:.3f}, rec: {recall_class_specific_finetuned[0]:.3f}, f1: {f1_class_specific_finetuned[0]:.3f}")
print(f"No: prec: {precision_class_specific_finetuned[1]:.3f}, rec: {recall_class_specific_finetuned[1]:.3f}, f1: {f1_class_specific_finetuned[1]:.3f}")