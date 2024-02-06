import pandas as pd
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# Load the files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')

# Replace labels with values expected by model
train.loc[train.target == 0, 'target'] = 2
train.loc[train.target == 1, 'target'] = 0

# Load the model and tokenizer
model_id = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3, device_map='auto')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

# Tokenize and split data
tokenized_train = Dataset.from_pandas(train).select_columns(['text', 'target']).rename_column('target', 'label').map(tokenize_function, batched=True)
train_data = tokenized_train.shuffle(seed=42).select(range(7000))
val_data = tokenized_train.shuffle(seed=42).select(range(7000, len(train)))

# Define training arguments
training_args = TrainingArguments(
    do_eval=True,
    evaluation_strategy='epoch',
    output_dir='test_trainer',
    logging_dir='test_trainer',
    logging_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=1,
    learning_rate=1e-05,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    load_best_model_at_end=True,
)

# Metrics function
def compute_metrics(p):
    metric=evaluate.load('f1')
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    return metric.compute(predictions=preds, references=labels, average="micro", pos_label=0)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

trainer.train()