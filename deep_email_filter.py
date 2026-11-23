

import argparse
import json
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


LABEL_MAP = {
    0: "Ham / Legitimate",
    1: "Spam",
    2: "Phishing",
}

TARGET_NAMES = ["ham", "spam", "phishing"]



# REPRODUCIBILITY


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# TEXT CLEANING

def clean_text(text):
   
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# DATA LOADING


def load_and_prepare_data(spam_csv, phishing_csv):
    spam_df = pd.read_csv(spam_csv)

    print("\nSpamAssasin columns:")
    print(spam_df.columns)

    spam_df["text"] = spam_df["subject"].fillna("") + " " + spam_df["body"].fillna("")

    spam_df["target"] = spam_df["label"].map({
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
        "ham": 0,
        "spam": 1,
    })

    spam_df = spam_df[["text", "target"]]
    spam_df.dropna(subset=["target"], inplace=True)

    phishing_df = pd.read_csv(phishing_csv)

    print("\nPhishing dataset columns:")
    print(phishing_df.columns)

    phishing_df["text"] = phishing_df["text_combined"].fillna("")

    phishing_df["target"] = phishing_df["label"].map({
        0: 0,
        1: 2,
        "0": 0,
        "1": 2,
        "ham": 0,
        "phishing": 2,
    })

    phishing_df = phishing_df[["text", "target"]]
    phishing_df.dropna(subset=["target"], inplace=True)

    df = pd.concat([spam_df, phishing_df], ignore_index=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 20]
    df["target"] = df["target"].astype(int)

    return df.reset_index(drop=True)



# TRAIN/TEST SPLIT + BALANCING


def create_better_training_and_test_sets(df, test_size=0.2, max_train_per_class=10000, seed=42):

    train_parts = []
    test_parts = []

    for label in [0, 1, 2]:
        class_df = df[df["target"] == label]
        train_c, test_c = train_test_split(
            class_df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        train_parts.append(train_c)
        test_parts.append(test_c)

    train_raw = pd.concat(train_parts, ignore_index=True)
    test_raw = pd.concat(test_parts, ignore_index=True)

    # Balanced training set
    balanced_train_parts = []
    for label in [0, 1, 2]:
        class_df = train_raw[train_raw["target"] == label]
        replace = len(class_df) < max_train_per_class
        sampled = class_df.sample(
            n=max_train_per_class,
            replace=replace,
            random_state=seed,
        )
        balanced_train_parts.append(sampled)

    train_df = pd.concat(balanced_train_parts, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Balanced test set for clear class-by-class evaluation
    min_test_count = test_raw["target"].value_counts().min()
    balanced_test_parts = []
    for label in [0, 1, 2]:
        sampled = test_raw[test_raw["target"] == label].sample(
            n=min_test_count,
            random_state=seed,
        )
        balanced_test_parts.append(sampled)

    test_df = pd.concat(balanced_test_parts, ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, test_df



# VOCABULARY + ENCODING


def build_vocab(texts, max_vocab_size=50000, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, count in counter.most_common(max_vocab_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len=300):
    tokens = str(text).split()
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_len]]

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))

    return ids


class EmailDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = encode_text(self.texts[idx], self.vocab, self.max_len)
        label = int(self.labels[idx])
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)



# DEEP LEARNING MODEL: TEXT CNN


class TextCNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim=128, num_classes=3, num_filters=128, filter_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(filter_sizes), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_x = torch.relu(conv(x))
            pooled_x = torch.max(conv_x, dim=2).values
            conv_outputs.append(pooled_x)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits



# TRAINING + EVALUATION


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / max(len(dataloader), 1), acc, np.array(all_labels), np.array(all_preds)


def predict_email(model, text, vocab, max_len, device):
    model.eval()
    cleaned = clean_text(text)
    encoded = encode_text(cleaned, vocab, max_len)

    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probabilities))

    return prediction, probabilities, cleaned




def save_model(model, vocab, config, model_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "config": config,
        "label_map": LABEL_MAP,
    }
    torch.save(checkpoint, model_path)


def load_saved_model(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        num_classes=config["num_classes"],
        num_filters=config["num_filters"],
        filter_sizes=tuple(config["filter_sizes"]),
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, vocab, config



# MAIN PROGRAM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spam_csv", default="SpamAssasin.csv")
    parser.add_argument("--phishing_csv", default="phishing_email.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--max_vocab_size", type=int, default=50000)
    parser.add_argument("--max_train_per_class", type=int, default=10000)
    parser.add_argument("--model_path", default="deep_email_filter.pt")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    df = load_and_prepare_data(args.spam_csv, args.phishing_csv)

    print("\nBefore training balance:")
    print(df["target"].value_counts().sort_index())

    train_df, test_df = create_better_training_and_test_sets(
        df,
        test_size=0.2,
        max_train_per_class=args.max_train_per_class,
        seed=42,
    )

    print("\nTraining data after balancing:")
    print(train_df["target"].value_counts().sort_index())

    print("\nBalanced test data:")
    print(test_df["target"].value_counts().sort_index())

    #small validation split from training data
    X_train, X_val, y_train, y_val = train_test_split(
        train_df["text"],
        train_df["target"],
        test_size=0.1,
        random_state=42,
        stratify=train_df["target"],
    )

    X_test = test_df["text"]
    y_test = test_df["target"]

    print("\nBuilding vocabulary...")
    vocab = build_vocab(X_train, max_vocab_size=args.max_vocab_size, min_freq=2)
    print("Vocabulary size:", len(vocab))

    train_dataset = EmailDataset(X_train, y_train, vocab, args.max_len)
    val_dataset = EmailDataset(X_val, y_val, vocab, args.max_len)
    test_dataset = EmailDataset(X_test, y_test, vocab, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    config = {
        "max_len": args.max_len,
        "max_vocab_size": args.max_vocab_size,
        "embedding_dim": 128,
        "num_classes": 3,
        "num_filters": 128,
        "filter_sizes": [3, 4, 5],
        "dropout": 0.5,
    }

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        num_classes=config["num_classes"],
        num_filters=config["num_filters"],
        filter_sizes=tuple(config["filter_sizes"]),
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining Deep Learning TextCNN...")
    best_val_acc = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_loss, test_acc, true_labels, predictions = evaluate_model(model, test_loader, criterion, device)

    print("Deep Learning TextCNN")
    print("Accuracy:", test_acc)

    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        predictions,
        target_names=TARGET_NAMES,
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))

    save_model(model, vocab, config, args.model_path)
    print("\nModel saved successfully.")
    print("Saved:", args.model_path)

    sample_email = """
    Philip Reynolds wrote:
>>Does anyone know how to list the biggest file in my
>>root directory?or the second biggest ..etc...
> $ find /dir -name \*  | xargs du -s | sort -n

You might want to put a '-type f' on that find.

Paul.


-- 
Irish Linux Users' Group: ilug@linux.ie
http://www.linux.ie/mailman/listinfo/ilug for (un)subscription information.
List maintainer: listmaster@linux.ie
    """

    pred, probs, cleaned = predict_email(model, sample_email, vocab, args.max_len, device)

    print("\nSample Email Prediction:")
    print(LABEL_MAP[pred])

    print("\nClass Probabilities:")
    for i, prob in enumerate(probs):
        print(f"{LABEL_MAP[i]}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
