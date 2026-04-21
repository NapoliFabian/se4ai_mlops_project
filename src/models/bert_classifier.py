import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
    
def load_data_from_csv(train_path, test_path, text_col="text", label_col="label"):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_texts = train_df[text_col].fillna("").tolist()
    train_labels = train_df[label_col].tolist()

    test_texts = test_df[text_col].fillna("").tolist()
    test_labels = test_df[label_col].tolist()

    return train_texts, train_labels, test_texts, test_labels   


def train_bert_classifier(
    train_csv,
    test_csv,
    model_dir="models/",
    epochs=2,
    batch_size=8,
    lr=2e-4
):

    os.makedirs(model_dir, exist_ok=True)


    train_texts, train_labels, _, _ = load_data_from_csv(
        train_csv,
        test_csv
    )

    # TOKENIZER + MODEL
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        ignore_mismatched_sizes=True  # 🔥 CRUCIALE
    )


    dataset = TextDataset(train_texts, train_labels, tokenizer)

    # TRAINING CONFIG
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # TRAIN
    trainer.train()

    # SAVE
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"[BERT] trained and saved in {model_dir}")

    return model, tokenizer



def predict_bert(model, tokenizer, texts, max_len=512, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", unit="batch"):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

        preds.extend(batch_preds)

    return preds