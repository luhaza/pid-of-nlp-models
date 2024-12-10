#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq)
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, load_dataset
# from peft import LoraConfig, get_peft_model


# In[10]:


import sys
import os

data_path = os.path.abspath(os.path.join(os.getcwd(), '../../data'))
sys.path.append(data_path)


# In[11]:


from data import train_set, test_set, sample_dataset, dataset

print(f"Size of test set: {len(test_set)}, size of train set: {len(train_set)}, no overlap: {len(train_set)+len(test_set)==len(dataset)}, size of sample (validation) set: {len(sample_dataset)}")


# In[15]:


type(train_set)


# In[16]:


def format_with_instruction(row):
    return {
        "instruction": "Classify the following sentence as liberal, neutral, or conservative.",
        "sentence": row["sentence"],
        "label": row["label"],
        "formatted_input": f"Instruction: Classify the following sentence as liberal, neutral, or conservative.\n"
                           f"Sentence: {row['sentence']}\n"
                           f"Options: liberal, neutral, conservative"
    }

train_data = [format_with_instruction(row) for row in train_set]
test_data = [format_with_instruction(row) for row in test_set]

train_ds = Dataset.from_pandas(pd.DataFrame(data=train_data))
test_ds = Dataset.from_pandas(pd.DataFrame(data=test_data))


# In[ ]:


# pass the token + load model
token = "token"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(example):
    return tokenizer(
        example["formatted_input"],
        truncation=True
    )


# In[ ]:


tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_test = test_ds.map(preprocess_function, batched=True)

tokenized_train


# In[41]:


model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.2-1B", num_labels=3, use_auth_token=token
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


# In[22]:


# # Check a sample after preprocessing to make sure everything is correct
# sample = train_ds[50]  # Check the first sample
# print("Sample after preprocessing:", sample)

# # Check if the padding token is applied correctly
# print("Padding token:", tokenizer.pad_token)


# In[ ]:





# In[42]:


import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# In[ ]:


training_args = TrainingArguments(
    output_dir="pid-ft-llama",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    group_by_length=True,
)


# In[52]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# In[53]:


# Train the model
trainer.train()


# In[ ]:


# Evaluate the model
# results = trainer.evaluate()

