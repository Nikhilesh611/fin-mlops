# save as finbert_train_gpu.py
import os
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

if __name__ == '__main__':   
# --- Basic checks / recommended settings
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True  # optional: can speed up conv-like ops

    # --- Config
    MODEL_NAME = "ProsusAI/finbert"
    OUTPUT_DIR = "./finbert_results_gpu"
    MODEL_ARTIFACTS_PATH = "./model_artifacts"
    NUM_LABELS = 3  # Positive, Negative, Neutral

    # --- Metrics
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = accuracy_score(p.label_ids, preds)
        f1 = f1_score(p.label_ids, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}

    # --- 1) Load dataset from Hugging Face Datasets Hub
    # We'll use the "financial_phrasebank" dataset (commonly used for fin-sentiment).
    # If that dataset is not available, replace load_dataset with your CSV loading code.
    # dataset = load_dataset("financial_phrasebank", "sentences_allagree")  # may download the dataset
    dataset = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)

    # The dataset has column 'sentence' and 'label' (label 0/1/2 or similar depending split)
    # Inspect dataset features:
    print(dataset)

    # Convert to train/test if dataset already has splits or use its train split and make a validation split
    if "train" in dataset:
        ds = dataset["train"]
    else:
        ds = dataset["validation"]  # fallback

    # We'll create an 80/20 train/validation split
    ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]

    # Inspect columns to find text & label column names
    print("Columns:", train_ds.column_names)
    # Usually: 'sentence' and 'label' — adjust below if different.

    # --- 2) Tokenizer and tokenization function
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_fn(batch):
        # change 'sentence' to whichever column holds text
        return tokenizer(batch["sentence"], truncation=True, padding=False)  # padding handled by data collator

    # Map tokenization (batched)
    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True)

    # Rename target column to "labels"
    if "label" in tokenized_train.column_names:
        tokenized_train = tokenized_train.rename_column("label", "labels")
        tokenized_eval = tokenized_eval.rename_column("label", "labels")
    else:
        raise ValueError("Couldn't find 'label' column in dataset")

    # Set format to PyTorch tensors when Trainer will consume them (Trainer + DataCollator will handle conversion)
    # but we won't set_format to "torch" now because we are using DataCollatorWithPadding which expects lists.
    # tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # --- 3) Model (load from Hub, override num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    # Optionally set label2id/id2label on model config for clearer saved config
    model.config.label2id = {"POSITIVE": 0, "NEGATIVE": 1, "NEUTRAL": 2}
    model.config.id2label = {v:k for k,v in model.config.label2id.items()}

    # --- 4) Data collator (dynamic padding improves memory efficiency)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 5) Training arguments tuned for GPU
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,   # try 8, increase if you have memory
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,                       # IMPORTANT: enable mixed precision for NVIDIA GPUs
        gradient_accumulation_steps=2,   # increase if you want larger effective batch
        dataloader_num_workers=4,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    # --- 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 7) Train
    print("Starting training on device:", "cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()

    # --- 8) Save artifacts
    model.save_pretrained(MODEL_ARTIFACTS_PATH)
    tokenizer.save_pretrained(MODEL_ARTIFACTS_PATH)
    print("Saved to", MODEL_ARTIFACTS_PATH)
