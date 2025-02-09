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

tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert/distilroberta-base', truncation_side="left")
boolq_dataset = load_dataset('google/boolq')
emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=510)

data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

train_dataset_boolq = boolq_dataset['train'].map(
    lambda e: {'text': f"{e['passage']}\n{e['question']}?\n"},
    remove_columns=['passage', 'question']
)
train_dataset_boolq = train_dataset_boolq.map(tokenize_function, batched=True)
train_dataset_boolq.set_format(type='torch', columns=['input_ids', 'attention_mask', 'answer'])
train_loader_boolq = torch.utils.data.DataLoader(train_dataset_boolq, shuffle=True, batch_size=4, collate_fn=data_collator)

validation_dataset_boolq = boolq_dataset['validation'].map(
    lambda e: {'text': f"{e['passage']}\n{e['question']}?\n"},
    remove_columns=['passage', 'question']
)
validation_dataset_boolq = validation_dataset_boolq.map(tokenize_function, batched=True)
validation_dataset_boolq.set_format(type='torch', columns=['input_ids', 'attention_mask', 'answer'])
validation_loader_boolq = torch.utils.data.DataLoader(validation_dataset_boolq, shuffle=True, batch_size=4, collate_fn=data_collator)

train_dataset_emo = emo_dataset['train']
train_dataset_emo = train_dataset_emo.map(tokenize_function, batched=True)
train_dataset_emo.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_loader_emo = torch.utils.data.DataLoader(train_dataset_emo, shuffle=True, batch_size=4, collate_fn=data_collator)

validation_dataset_emo = emo_dataset['validation']
validation_dataset_emo = validation_dataset_emo.map(tokenize_function, batched=True)
validation_dataset_emo.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
validation_loader_emo = torch.utils.data.DataLoader(validation_dataset_emo, shuffle=True, batch_size=4, collate_fn=data_collator)

test_dataset_emo = emo_dataset['test']
test_dataset_emo = test_dataset_emo.map(tokenize_function, batched=True)
test_dataset_emo.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_loader_emo = torch.utils.data.DataLoader(test_dataset_emo, shuffle=True, batch_size=4, collate_fn=data_collator)

## 3.1 Task fine-tune distilRoberta for boolQ and compute accuracy
########################

model31 = transformers.RobertaForSequenceClassification.from_pretrained('distilbert/distilroberta-base')
model31.classifier = torch.nn.Linear(in_features=768, out_features=1)

model31.to(device)

optimizer = torch.optim.AdamW(model31.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8)

num_epochs = 1
loss_values = []
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(num_epochs):
    model31.train()
    for batch in tqdm.tqdm(train_loader_boolq):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model31(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_predictions = outputs.logits[:, -1]
        loss = loss_fn(last_predictions.squeeze(-1), batch['answer'].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_values.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

plt.plot(loss_values)
plt.title('Training loss over time')
plt.xlabel('Step')
plt.ylabel('Loss')
# plt.show()
plt.savefig('31_BoolQ.png')
plt.close()

model31.eval()

predictions = []
true_labels = []
for batch in tqdm.tqdm(validation_loader_boolq):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model31(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits[:, -1].squeeze(-1).detach().cpu().numpy()
    label_ids = batch['answer'].to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.where(torch.sigmoid(torch.tensor(flat_predictions)).numpy() >= 0.5, 1, 0)
flat_true_labels = np.concatenate(true_labels, axis=0)

accuracy_31 = sklearn.metrics.accuracy_score(flat_true_labels, flat_predictions)
_, _, f1_31, _ = sklearn.metrics.precision_recall_fscore_support(flat_true_labels, flat_predictions, average='macro')

precision_class_specific_31, recall_class_specific_31, f1_class_specific_31, _ = sklearn.metrics.precision_recall_fscore_support(flat_true_labels, flat_predictions, labels=[0, 1])

print("\nCheckpoint 3.1:")
print(f"Overall: acc: {accuracy_31:.3f}, f1: {f1_31:.3f}")
print(f"Yes: prec: {precision_class_specific_31[0]:.3f}, rec: {recall_class_specific_31[0]:.3f}, f1: {f1_class_specific_31[0]:.3f}")
print(f"No: prec: {precision_class_specific_31[1]:.3f}, rec: {recall_class_specific_31[1]:.3f}, f1: {f1_class_specific_31[1]:.3f}")


## 3.2 Task fine-tune distilRoberta for emoBank-valence and compute accuracy
########################

model32 = transformers.RobertaForSequenceClassification.from_pretrained('distilbert/distilroberta-base')
model32.classifier = torch.nn.Linear(in_features=768, out_features=1)

model32.to(device)

optimizer = torch.optim.AdamW(model32.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8)

num_epochs = 1
loss_values = []
loss_fn = torch.nn.MSELoss()
for epoch in range(num_epochs):
    model32.train()
    for batch in tqdm.tqdm(train_loader_emo):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model32(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_predictions = outputs.logits[:, -1]
        loss = loss_fn(last_predictions.squeeze(-1), batch['labels'].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_values.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

plt.plot(loss_values)
plt.title('Training Loss over time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('32_Emo.png')
plt.close()


model32.eval()
predictions = []
true_labels = []
for batch in tqdm.tqdm(validation_loader_emo):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model32(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits[:, -1].squeeze(-1).detach().cpu().numpy()
    label_ids = batch['labels'].to('cpu').numpy()
    predictions.extend(logits)
    true_labels.extend(label_ids)

predictions = np.array(predictions)
true_labels = np.array(true_labels)
mae32_val = sklearn.metrics.mean_absolute_error(true_labels, predictions)
r32_val = np.corrcoef(predictions, true_labels)[0, 1]


predictions = []
true_labels = []
for batch in tqdm.tqdm(test_loader_emo):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model32(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits[:, -1].squeeze(-1).detach().cpu().numpy()
    label_ids = batch['labels'].to('cpu').numpy()
    predictions.extend(logits)
    true_labels.extend(label_ids)

predictions = np.array(predictions)
true_labels = np.array(true_labels)
mae32_test = sklearn.metrics.mean_absolute_error(true_labels, predictions)
r32_test = np.corrcoef(predictions, true_labels)[0, 1]

print("\nCheckpoint 3.2:")
print(f"Validation: mae: {mae32_val:.3f}, r: {r32_val:.3f}")
print(f"  Test: mae: {mae32_test:.3f}, r: {r32_test:.3f}")

## 3.3 Make modifications to transformer distilRoberta and fine-tune the LM
########################

model331_boolq = transformers.RobertaForSequenceClassification.from_pretrained('distilbert/distilroberta-base')
model332_boolq = transformers.RobertaForSequenceClassification.from_pretrained('distilbert/distilroberta-base')
model333_boolq = transformers.RobertaForSequenceClassification.from_pretrained('distilbert/distilroberta-base')

model331_emo = transformers.RobertaForSequenceClassification.from_pretrained('distilbert/distilroberta-base')
model332_emo = transformers.RobertaForSequenceClassification.from_pretrained('distilbert/distilroberta-base')
model333_emo = transformers.RobertaForSequenceClassification.from_pretrained('distilbert/distilroberta-base')

model331_boolq.to(device)
model332_boolq.to(device)
model333_boolq.to(device)

model331_emo.to(device)
model332_emo.to(device)
model333_emo.to(device)


def randomize_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch.rand(module.weight.size())
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1)
    model.classifier = torch.nn.Linear(in_features=768, out_features=1)
    return model

model331_boolq = randomize_weights(model331_boolq)
model331_emo = randomize_weights(model331_emo)

model331_boolq.to(device)
model331_emo.to(device)

## 3.3.1 distilRB-rand with BoolQ
########################


# train_dataset = boolq_dataset['train'].map(
#     lambda e: {'text': f"{e['passage']}\n{e['question']}?\n"},
#     remove_columns=['passage', 'question']
# )
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'answer'])
# data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

optimizer = torch.optim.AdamW(model331_boolq.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8)

num_epochs = 1
loss_values = []
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(num_epochs):
    model331_boolq.train()
    for batch in tqdm.tqdm(train_loader_boolq):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model331_boolq(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_predictions = outputs.logits[:, -1]
        loss = loss_fn(last_predictions.squeeze(-1), batch['answer'].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_values.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

plt.plot(loss_values)
plt.title('Training loss over time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig("331_BoolQ.png")
plt.close()

## 3.3.1 distilRB-rand with EmoBank
########################

# train_dataset = emo_dataset['train']
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
# data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

optimizer = torch.optim.AdamW(model331_emo.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8)

num_epochs = 1
loss_values = []
loss_fn = torch.nn.MSELoss()
for epoch in range(num_epochs):
    model331_emo.train()
    for batch in tqdm.tqdm(train_loader_emo):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model331_emo(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_predictions = outputs.logits[:, -1]
        loss = loss_fn(last_predictions.squeeze(-1), batch['labels'].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_values.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

plt.plot(loss_values)
plt.title('Training loss over time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('331_Emo.png')
plt.close()


## 3.3.2 distilRB-KQ
########################

def average_KQ(model):
    for i in range(4, 6):
        key_layer = model.roberta.encoder.layer[i].attention.self.key
        query_layer = model.roberta.encoder.layer[i].attention.self.query
        mean_weight = (key_layer.weight.data + query_layer.weight.data) / 2
        key_layer.weight.data = mean_weight
        query_layer.weight.data = mean_weight
    model.classifier = torch.nn.Linear(in_features=768, out_features=1)
    return model

model332_boolq = average_KQ(model332_boolq)
model332_emo = average_KQ(model332_emo)

model332_boolq.to(device)
model332_emo.to(device)


## 3.3.2 distilRB-KQ with BoolQ
########################

# train_dataset = boolq_dataset['train'].map(
#     lambda e: {'text': f"{e['passage']}\n{e['question']}?\n"},
#     remove_columns=['passage', 'question']
# )

# train_dataset = train_dataset.map(tokenize_function, batched=True)
# train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'answer'])
# data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

optimizer = torch.optim.AdamW(model332_boolq.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8)

num_epochs = 1
loss_values = []
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(num_epochs):
    model332_boolq.train()
    for batch in tqdm.tqdm(train_loader_boolq):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model332_boolq(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_predictions = outputs.logits[:, -1]
        loss = loss_fn(last_predictions.squeeze(-1), batch['answer'].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_values.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

plt.plot(loss_values)
plt.title('Training loss over time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig("332_BoolQ.png")
plt.close()

## 3.3.2 distilRB-KQ with EmoValence
########################

# train_dataset = emo_dataset['train']
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
# data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

optimizer = torch.optim.AdamW(model332_emo.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8)

num_epochs = 1
loss_values = []
loss_fn = torch.nn.MSELoss()
for epoch in range(num_epochs):
    model332_emo.train()
    for batch in tqdm.tqdm(train_loader_emo):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model332_emo(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_predictions = outputs.logits[:, -1]
        loss = loss_fn(last_predictions.squeeze(-1), batch['labels'].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_values.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

plt.plot(loss_values)
plt.title('Training loss over time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('332_Emo.png')
plt.close()

## 3.3.3 distilRB-nores
########################

class RobertaOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class RobertaSelfOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


config = transformers.RobertaConfig.from_pretrained('distilbert/distilroberta-base')

roberta_obj_boolq1 = RobertaOutput(config)
roberta_obj_boolq2 = RobertaOutput(config)
roberta_obj_emo1 = RobertaOutput(config)
roberta_obj_emo2 = RobertaOutput(config)

roberta_obj_boolq1.dense.weight.data = model333_boolq.roberta.encoder.layer[-1].output.dense.weight.data
roberta_obj_boolq2.dense.weight.data = model333_boolq.roberta.encoder.layer[-2].output.dense.weight.data
roberta_obj_emo1.dense.weight.data = model333_emo.roberta.encoder.layer[-1].output.dense.weight.data
roberta_obj_emo2.dense.weight.data = model333_emo.roberta.encoder.layer[-2].output.dense.weight.data

model333_boolq.roberta.encoder.layer[-1].output = roberta_obj_boolq1
model333_boolq.roberta.encoder.layer[-2].output = roberta_obj_boolq2
model333_emo.roberta.encoder.layer[-1].output = roberta_obj_emo1
model333_emo.roberta.encoder.layer[-2].output = roberta_obj_emo2

roberta_self_obj_boolq1 = RobertaSelfOutput(config)
roberta_self_obj_boolq2 = RobertaSelfOutput(config)
roberta_self_obj_emo1 = RobertaSelfOutput(config)
roberta_self_obj_emo2 = RobertaSelfOutput(config)

roberta_self_obj_boolq1.dense.weight.data = model333_boolq.roberta.encoder.layer[-1].attention.output.dense.weight.data
roberta_self_obj_boolq2.dense.weight.data = model333_boolq.roberta.encoder.layer[-2].attention.output.dense.weight.data
roberta_self_obj_emo1.dense.weight.data = model333_emo.roberta.encoder.layer[-1].attention.output.dense.weight.data
roberta_self_obj_emo2.dense.weight.data = model333_emo.roberta.encoder.layer[-2].attention.output.dense.weight.data

model333_boolq.roberta.encoder.layer[-1].attention.output = roberta_self_obj_boolq1
model333_boolq.roberta.encoder.layer[-2].attention.output = roberta_self_obj_boolq2
model333_emo.roberta.encoder.layer[-1].attention.output = roberta_self_obj_emo1
model333_emo.roberta.encoder.layer[-2].attention.output = roberta_self_obj_emo2

## 3.3.3 distilRB-nores with BoolQ
########################

model333_boolq.to(device)
optimizer = torch.optim.AdamW(model333_boolq.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8)

num_epochs = 1
loss_values = []
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(num_epochs):
    model333_boolq.train()
    for batch in tqdm.tqdm(train_loader_boolq):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model333_boolq(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_predictions = outputs.logits[:, -1]
        loss = loss_fn(last_predictions.squeeze(-1), batch['answer'].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_values.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

plt.plot(loss_values)
plt.title('Training loss over time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig("333_BoolQ.png")
plt.close()

## 3.3.3 distilRB-nores with EmoValence
########################

model333_emo.to(device)
optimizer = torch.optim.AdamW(model333_emo.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8)

num_epochs = 1
loss_values = []
loss_fn = torch.nn.MSELoss()
for epoch in range(num_epochs):
    model333_emo.train()
    for batch in tqdm.tqdm(train_loader_emo):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model333_emo(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_predictions = outputs.logits[:, -1]
        loss = loss_fn(last_predictions.squeeze(-1), batch['labels'].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_values.append(loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed.")

plt.plot(loss_values)
plt.title('Training loss over time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('333_Emo.png')
plt.close()

## 3.4 Task fine-tune and recompute accuracies on tasks
########################

# validation_dataset = boolq_dataset['validation'].map(
#     lambda e: {'text': f"{e['passage']}\n{e['question']}?\n"},
#     remove_columns=['passage', 'question']
# )
# validation_dataset = validation_dataset.map(tokenize_function, batched=True)
# validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'answer'])
# validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

# test_dataset = emo_dataset['test']
# test_dataset = test_dataset.map(tokenize_function, batched=True)
# test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
# test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

## Roberta Random with BoolQ
########################

model331_boolq.eval()
predictions = []
true_labels = []
for batch in tqdm.tqdm(validation_loader_boolq):
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model331_boolq(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

    logits = outputs.logits[:, -1].squeeze(-1).detach().cpu().numpy()
    label_ids = batch['answer'].to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.where(torch.sigmoid(torch.tensor(flat_predictions)).numpy() >= 0.5, 1, 0)
flat_true_labels = np.concatenate(true_labels, axis=0)

accuracy_331 = sklearn.metrics.accuracy_score(flat_true_labels, flat_predictions)
_, _, f1_331, _ = sklearn.metrics.precision_recall_fscore_support(flat_true_labels, flat_predictions, average='macro')

## Roberta Random with EmoBank
########################

model331_emo.eval()
predictions = []
true_labels = []
for batch in tqdm.tqdm(test_loader_emo):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model331_emo(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits[:, -1].squeeze(-1).detach().cpu().numpy()
    label_ids = batch['labels'].to('cpu').numpy()
    predictions.extend(logits)
    true_labels.extend(label_ids)

predictions = np.array(predictions)
true_labels = np.array(true_labels)
mae331 = sklearn.metrics.mean_absolute_error(true_labels, predictions)
r331 = np.corrcoef(predictions, true_labels)[0, 1]


## Roberta KQ Mean with BoolQ
########################

model332_boolq.eval()
predictions = []
true_labels = []
for batch in tqdm.tqdm(validation_loader_boolq):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model332_boolq(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits[:, -1].squeeze(-1).detach().cpu().numpy()
    label_ids = batch['answer'].to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.where(torch.sigmoid(torch.tensor(flat_predictions)).numpy() >= 0.5, 1, 0)
flat_true_labels = np.concatenate(true_labels, axis=0)

accuracy_332 = sklearn.metrics.accuracy_score(flat_true_labels, flat_predictions)
_, _, f1_332, _ = sklearn.metrics.precision_recall_fscore_support(flat_true_labels, flat_predictions, average='macro')

## Roberta KQ Mean with EmoBank
########################

model332_emo.eval()
predictions = []
true_labels = []
for batch in tqdm.tqdm(test_loader_emo):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model332_emo(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits[:, -1].squeeze(-1).detach().cpu().numpy()
    label_ids = batch['labels'].to('cpu').numpy()
    predictions.extend(logits)
    true_labels.extend(label_ids)

predictions = np.array(predictions)
true_labels = np.array(true_labels)
mae332 = sklearn.metrics.mean_absolute_error(true_labels, predictions)
r332 = np.corrcoef(predictions, true_labels)[0, 1]


## Roberta No Res Mean with BoolQ
########################

model333_boolq.eval()
predictions = []
true_labels = []
for batch in tqdm.tqdm(validation_loader_boolq):
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model333_boolq(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

    logits = outputs.logits[:, -1].squeeze(-1).detach().cpu().numpy()
    label_ids = batch['answer'].to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.where(torch.sigmoid(torch.tensor(flat_predictions)).numpy() >= 0.5, 1, 0)
flat_true_labels = np.concatenate(true_labels, axis=0)

accuracy_333 = sklearn.metrics.accuracy_score(flat_true_labels, flat_predictions)
_, _, f1_333, _ = sklearn.metrics.precision_recall_fscore_support(flat_true_labels, flat_predictions, average='macro')

## Roberta No Res Mean with EmoBank
########################

model333_emo.eval()
predictions = []
true_labels = []
for batch in tqdm.tqdm(test_loader_emo):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model333_emo(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits[:, -1].squeeze(-1).detach().cpu().numpy()
    label_ids = batch['labels'].to('cpu').numpy()
    predictions.extend(logits)
    true_labels.extend(label_ids)

predictions = np.array(predictions)
true_labels = np.array(true_labels)
mae333 = sklearn.metrics.mean_absolute_error(true_labels, predictions)
r333 = np.corrcoef(predictions, true_labels)[0, 1]

print("\nCheckpoint: 3.4:")
print(f"boolq validation set:")
print(f"distilroberta: overall acc: {accuracy_31:.3f}, f1: {f1_31:.3f}")
print(f"distilRB-rand: overall acc: {accuracy_331:.3f}, f1: {f1_331:.3f}")
print(f"distilRB-KQ: overall acc: {accuracy_332:.3f}, f1: {f1_332:.3f}")
print(f"distilRB-nores: overall acc: {accuracy_333:.3f}, f1: {f1_333:.3f}")
print(f"emobank test set:")
print(f"distilroberta: mae: {mae32_test:.3f}, r: {r32_test:.3f}")
print(f"distilRB-rand: mae: {mae331:.3f}, r: {r331:.3f}")
print(f"distilRB-KQ: mae: {mae332:.3f}, r: {r332:.3f}")
print(f"distilRB-nores: mae: {mae333:.3f}, r: {r333:.3f}")
