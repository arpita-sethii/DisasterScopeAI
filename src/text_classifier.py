
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

LABEL_MAP = {"earthquake": 0, "flood": 1, "wildfire": 2, "hurricane": 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def clean_text(text):
    if pd.isna(text):
        return ""
    s = str(text)
    s = re.sub(r"http\S+|www\S+", "", s)
    s = re.sub(r"@\w+", "", s)
    s = re.sub(r"#(\w+)", r"\1", s)
    s = s.replace("&amp;", "and")
    return s.strip()

def map_disaster(text):
    t = text.lower()
    if any(k in t for k in ["earthquake", "quake", "seismic", "richter"]):
        return "earthquake"
    if any(k in t for k in ["flood", "submerged", "inundation", "flooding"]):
        return "flood"
    if any(k in t for k in ["fire", "wildfire", "burning", "blaze"]):
        return "wildfire"
    if any(k in t for k in ["hurricane", "cyclone", "typhoon", "storm", "landfall"]):
        return "hurricane"
    return None

def prepare_dataset(csv_path, filter_target=True, min_len=10, test_size=0.2, random_state=42):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    df['text'] = df['text'].apply(clean_text)
    if filter_target and 'target' in df.columns:
        df = df[df['target'] == 1]
    df['disaster'] = df['text'].apply(map_disaster)
    df = df.dropna(subset=['disaster']).reset_index(drop=True)
    df['label'] = df['disaster'].map(LABEL_MAP)
    df = df[df['text'].str.len() >= min_len].drop_duplicates(subset=['text']).reset_index(drop=True)
    if df['label'].nunique() < 2:
        raise ValueError("Not enough classes after mapping. Check dataset or mapping rules.")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=test_size, stratify=df['label'], random_state=random_state
    )
    return train_texts, test_texts, train_labels, test_labels, df

def tokenize_dataset(tokenizer, texts_labels):
    texts, labels = texts_labels
    ds = Dataset.from_dict({"text": texts, "label": labels})
    def tok(ex):
        return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=128)
    ds = ds.map(tok, batched=True)
    ds = ds.remove_columns(["text"]).with_format("torch")
    return ds

def plot_and_save_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.title("Confusion Matrix - Text Classifier")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return out_path

def train_pipeline(csv_path, output_dir, device):
    print("Preparing dataset...")
    train_texts, test_texts, train_labels, test_labels, df = prepare_dataset(csv_path)
    print(f"Training samples: {len(train_texts)} | Test samples: {len(test_texts)}")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = tokenize_dataset(tokenizer, (train_texts, train_labels))
    test_ds  = tokenize_dataset(tokenizer, (test_texts, test_labels))
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(LABEL_MAP)).to(device)
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20,
            load_best_model_at_end=True
        )
    except TypeError:
        training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=3, per_device_train_batch_size=16, logging_steps=20)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics)
    print("Starting training...")
    trainer.train()
    eval_res = trainer.evaluate()
    print("Eval results:", eval_res)
    
    pred_output = trainer.predict(test_ds)
    preds = np.argmax(pred_output.predictions, axis=1)
    report = classification_report(test_labels, preds, target_names=list(LABEL_MAP.keys()), output_dict=True)
    print("Classification report:\n", classification_report(test_labels, preds, target_names=list(LABEL_MAP.keys())))
   
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "confusion_matrix_text.png")
    plot_and_save_confusion_matrix(test_labels, preds, list(LABEL_MAP.keys()), cm_path)
    
    history = {
        "eval_results": {k: float(v) for k,v in eval_res.items()},
        "classification_report": report,
        "num_train": len(train_texts),
        "num_test": len(test_texts)
    }
    with open(os.path.join(output_dir, "training_history_text.json"), "w") as f:
        json.dump(history, f, indent=2)
   
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Saved model and artifacts to:", output_dir)
    return output_dir

def load_model_and_predict(model_dir, texts, device):
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            for p, conf in zip(preds, probs):
                outputs.append({"label": INV_LABEL_MAP[int(p)], "confidence": float(conf[int(p)])})
    return outputs


def keyword_disaster_classifier(text):
    t = text.lower()
    if any(k in t for k in ["earthquake", "quake", "seismic", "richter"]):
        return "earthquake", 0.9
    if any(k in t for k in ["flood", "submerged", "inundation", "flooding"]):
        return "flood", 0.9
    if any(k in t for k in ["fire", "wildfire", "burning", "blaze"]):
        return "wildfire", 0.9
    if any(k in t for k in ["hurricane", "cyclone", "typhoon", "storm", "landfall"]):
        return "hurricane", 0.9
    return "unknown", 0.25

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict", "demo"], default="demo")
    parser.add_argument("--data", type=str, default="data/train.csv", help="path to local CSV (must contain 'text').")
    parser.add_argument("--output", type=str, default="models/distilbert_disaster_classifier", help="model output dir")
    parser.add_argument("--model_dir", type=str, default="models/distilbert_disaster_classifier", help="model dir to load for predict")
    parser.add_argument("--text", type=str, default=None, help="single text for prediction (predict mode)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.mode == "train":
        print("TRAIN MODE")
        out = train_pipeline(args.data, args.output, device)
        print("Training complete. Artifacts saved to:", out)
        return

    if args.mode == "predict":
        texts = [args.text] if args.text else [
            "Massive earthquake hits Tokyo, buildings collapsed!",
            "Flash flood warning for Houston, roads submerged!",
            "Wildfire spreading near Los Angeles, evacuations ordered!"
        ]
        if not os.path.exists(args.model_dir):
            print("Model folder not found at", args.model_dir)
            print("You can still use the keyword fallback classifier.")
            for t in texts:
                lab, conf = keyword_disaster_classifier(t)
                print(f"[KW] {t} -> {lab} ({conf})")
            return

        preds = load_model_and_predict(args.model_dir, texts, device)
        for t, p in zip(texts, preds):
            print("TEXT:", t)
            print("PRED:", p["label"], " CONF:", round(p["confidence"], 4))
        return

if __name__ == "__main__":
    main()
