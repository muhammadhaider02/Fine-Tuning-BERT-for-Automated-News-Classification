# ================================================================
# PART 1 — INSTALL, IMPORTS, DATA LOADING, CLEANING, TOKENIZATION
#           + CLASS WEIGHTS
# ================================================================

!pip install -q transformers datasets nltk tensorflow --quiet

import os
import re
import time
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorflow.keras.datasets import reuters as keras_reuters

import nltk
nltk.download("stopwords")

# ================================================================
# 1. Reproducibility & Device
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available. Please enable a T4 GPU in Colab before running.")

DEVICE = torch.device("cuda")
print("Using device:", DEVICE)

set_seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ================================================================
# 2. Load Reuters (Keras, NUM_WORDS=10000)
# ================================================================

NUM_WORDS = 10000
(train_x, train_y), (test_x, test_y) = keras_reuters.load_data(num_words=NUM_WORDS)

print(f"Raw train samples: {len(train_x)}")
print(f"Raw test  samples: {len(test_x)}")

# ================================================================
# 3. Decode indices -> text (SGM-style)
# ================================================================

word_index = keras_reuters.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "[PAD]"
reverse_word_index[1] = "[START]"
reverse_word_index[2] = "[UNK]"
reverse_word_index[3] = "[UNUSED]"

def decode_sequence(seq):
    return " ".join(reverse_word_index.get(i, "[UNK]") for i in seq)

train_texts_raw = [decode_sequence(seq) for seq in train_x]
test_texts_raw  = [decode_sequence(seq) for seq in test_x]

print("\nExample raw decoded train text:")
print(train_texts_raw[0][:300], "...")
print("Raw label:", int(train_y[0]))

# ================================================================
# 4. Minimal Text Cleaning (tuned for Small BERT uncased)
#    - keep stopwords, punctuation, numbers, repeats
#    - just lowercase + remove Keras artifacts
#    - drop very tiny docs (< 3 tokens)
# ================================================================

def clean_text(text: str) -> str:
    # Lowercase for consistency (Small BERT-base-uncased)
    text = text.lower()

    # Remove Keras artifact tokens only
    for token in ("[unk]", "[pad]", "[start]", "[unused]"):
        text = text.replace(token, " ")

    # Collapse extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Drop extremely short docs
    tokens = text.split()
    if len(tokens) < 3:
        return ""

    return text

train_texts_clean, train_labels_clean = [], []
for txt, lbl in zip(train_texts_raw, train_y):
    ct = clean_text(txt)
    if ct:
        train_texts_clean.append(ct)
        train_labels_clean.append(int(lbl))

test_texts_clean, test_labels_clean = [], []
for txt, lbl in zip(test_texts_raw, test_y):
    ct = clean_text(txt)
    if ct:
        test_texts_clean.append(ct)
        test_labels_clean.append(int(lbl))

print(f"\nAfter cleaning: {len(train_texts_clean)} train, {len(test_texts_clean)} test.")

# ================================================================
# 5. Remove ultra-rare classes and reindex labels
# ================================================================

RARE_THRESHOLD = 50

label_counter = Counter(train_labels_clean)
keep_labels = {lbl for lbl, cnt in label_counter.items() if cnt >= RARE_THRESHOLD}

print(f"\nOriginal num classes (raw): {len(label_counter)}")
print(f"Classes kept (>= {RARE_THRESHOLD} samples): {len(keep_labels)}")

train_texts_final = []
train_labels_final_old = []
for txt, lbl in zip(train_texts_clean, train_labels_clean):
    if lbl in keep_labels:
        train_texts_final.append(txt)
        train_labels_final_old.append(lbl)

test_texts_final = []
test_labels_final_old = []
for txt, lbl in zip(test_texts_clean, test_labels_clean):
    if lbl in keep_labels:
        test_texts_final.append(txt)
        test_labels_final_old.append(lbl)

sorted_labels = sorted(keep_labels)
old_to_new = {old: i for i, old in enumerate(sorted_labels)}

train_labels_final = [old_to_new[l] for l in train_labels_final_old]
test_labels_final  = [old_to_new[l] for l in test_labels_final_old]

num_classes = len(sorted_labels)

print(f"\nFinal num_classes:   {num_classes}")
print(f"Final train samples: {len(train_texts_final)}")
print(f"Final test  samples: {len(test_texts_final)}")

print("\nExample cleaned train text:")
print(train_texts_final[0][:300], "...")
print("Mapped label:", train_labels_final[0])

# For StratifiedKFold later
train_labels = train_labels_final

# ================================================================
# 6. Build HuggingFace Dataset
# ================================================================

train_ds = Dataset.from_dict({"text": train_texts_final, "label": train_labels_final})
test_ds  = Dataset.from_dict({"text": test_texts_final,  "label": test_labels_final})

dataset = DatasetDict({"train": train_ds, "test": test_ds})
print("\nDatasetDict structure:")
print(dataset)

# ================================================================
# 7. Tokenization (BERT-base-uncased, WordPiece, MAX_LEN=300)
# ================================================================

model_name = "google/bert_uncased_L-4_H-512_A-8"
print(f"\nLoading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_LEN = 300  # longer context, still fine for BERT-base

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

print("\nTokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_batch, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

print(tokenized_dataset)

# ================================================================
# 8. Class Weights (inverse frequency + clipping)
# ================================================================

class_counts = np.zeros(num_classes, dtype=np.float32)
for lbl in train_labels:
    class_counts[lbl] += 1

class_weights = 1.0 / np.maximum(class_counts, 1.0)
# Clip extreme weights to avoid exploding loss on rare classes
class_weights = np.clip(class_weights, 0.0, 10.0)
# Renormalize so average weight ≈ 1
class_weights = class_weights * (num_classes / class_weights.sum())

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

print("\nClass counts:", class_counts)
print("Class weights (clipped & normalized):", class_weights)


# ================================================================
# PART 2 — MODEL, FOCAL TRAINER, EARLY STOPPING, TRAIN-ONE-FOLD
# ================================================================

from transformers.trainer_utils import get_last_checkpoint

# ---------------------------------------------------------------
# 1. Model factory with native HF gradient checkpointing
# ---------------------------------------------------------------
def create_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,              # "google/bert_uncased_L-4_H-512_A-8"
        num_labels=num_classes,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.1,
    )
    # Native HF gradient checkpointing (saves GPU memory)
    model.gradient_checkpointing_enable()
    return model.to(DEVICE)


# ---------------------------------------------------------------
# 2. Focal-loss Trainer (class weights + label smoothing)
#    Signature matches Trainer.training_step:
#    compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)
# ---------------------------------------------------------------
class FocalTrainer(Trainer):
    def __init__(self, class_weights, gamma=0.5, smoothing=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.gamma = gamma
        self.smoothing = smoothing
        self._cw_dev = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # num_items_in_batch is ignored but must be accepted for compatibility
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, C)

        if self._cw_dev is None:
            self._cw_dev = self.class_weights.to(logits.device)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        n_classes = logits.size(-1)

        # Label smoothing target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / n_classes)
            true_dist.scatter_(
                1,
                labels.unsqueeze(1),
                1 - self.smoothing + self.smoothing / n_classes,
            )

        focal = (1.0 - probs) ** self.gamma
        alpha = self._cw_dev.unsqueeze(0)

        loss = -(true_dist * alpha * focal * log_probs).sum(dim=-1).mean()

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------
# 3. Metrics (accuracy + weighted precision/recall/F1)
# ---------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


# ---------------------------------------------------------------
# 4. Early Stopping + Best Model Saver
#    - logs VAL loss/metrics once per epoch in a table
# ---------------------------------------------------------------
class EarlyStopCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, save_path, patience=7, delta=1e-4):
        super().__init__()
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.save_path = save_path
        self.patience = patience
        self.delta = delta

        self.best_f1 = None
        self.counter = 0
        self._header_printed = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Print table header once
        if not self._header_printed:
            print("Epoch\tTraining Loss\tValidation Loss\tAccuracy\tPrecision\tRecall\tF1")
            self._header_printed = True

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)

        # We rely on this extra evaluate() call for clean table logging.
        # (HF will also do its own eval because eval_strategy='epoch'.)
        metrics = self.trainer.evaluate(self.eval_dataset)

        val_loss = metrics.get("eval_loss", None)
        val_acc  = metrics.get("eval_accuracy", None)
        val_prec = metrics.get("eval_precision", None)
        val_rec  = metrics.get("eval_recall", None)
        val_f1   = metrics.get("eval_f1", None)

        def fmt(x):
            return f"{x:.6f}" if isinstance(x, (float, int)) else "NA"

        # We don't track training loss here → column shows "No log"
        print(
            f"{epoch}\tNo log\t{fmt(val_loss)}\t{fmt(val_acc)}\t"
            f"{fmt(val_prec)}\t{fmt(val_rec)}\t{fmt(val_f1)}"
        )

        # Early stopping on val_f1
        if val_f1 is not None:
            if (self.best_f1 is None) or (val_f1 - self.best_f1 > self.delta):
                self.best_f1 = val_f1
                self.counter = 0
                print(
                    f"[EarlyStop] Epoch {epoch} — eval_f1={val_f1:.4f}\n"
                    f"  → New best model, saving to {self.save_path}"
                )
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(self.trainer.model.state_dict(), self.save_path)
            else:
                self.counter += 1
                print(
                    f"[EarlyStop] Epoch {epoch} — eval_f1={val_f1:.4f}\n"
                    f"  → No improvement ({self.counter}/{self.patience})"
                )
                if self.counter >= self.patience:
                    print("  → Early stopping triggered.")
                    control.should_training_stop = True

        return control


# ---------------------------------------------------------------
# 5. Train-one-fold helper
#    Uses eval_strategy='epoch' + auto-resume from last checkpoint
# ---------------------------------------------------------------
def train_one_fold(train_ds, val_ds, fold_dir, fold_name):

    print(f"\n========== TRAINING {fold_name} | LR=3e-05 ==========\n")

    model = create_model()

    training_args = TrainingArguments(
        output_dir=fold_dir,
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,  # effective batch size = 32
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,

        fp16=True,
        report_to="none",
        seed=42,
        disable_tqdm=False,
    )

    trainer = FocalTrainer(
        class_weights=class_weights_tensor,
        gamma=0.5,
        smoothing=0.05,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    best_model_path = os.path.join(fold_dir, "best_model.pt")

    early_stop_cb = EarlyStopCallback(
        trainer=trainer,
        eval_dataset=val_ds,
        save_path=best_model_path,
        patience=7,
        delta=1e-4,
    )
    trainer.add_callback(early_stop_cb)

    # -----------------------------
    # Auto-resume from last ckpt
    # -----------------------------
    last_checkpoint = None
    if os.path.isdir(fold_dir):
        last_checkpoint = get_last_checkpoint(fold_dir)

    if last_checkpoint is not None:
        print(f"Resuming from last checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    return best_model_path


# ================================================================
# PART 3 — 3-FOLD STRATIFIED CROSS-VALIDATION
# ================================================================

print("\n========== 3-FOLD STRATIFIED CROSS-VALIDATION ==========\n")

indices = np.arange(len(train_labels))
labels_array = np.array(train_labels)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

base_output_dir = "bert_cv_runs"
os.makedirs(base_output_dir, exist_ok=True)

fold_best_paths = []
fold_val_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels_array), start=1):

    print(f"\n==================== FOLD {fold_idx} / 3 ====================\n")

    # Fold-specific subset
    train_fold = tokenized_dataset["train"].select(train_idx.tolist())
    val_fold   = tokenized_dataset["train"].select(val_idx.tolist())

    fold_dir = os.path.join(base_output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # Train fold
    best_path = train_one_fold(
        train_ds=train_fold,
        val_ds=val_fold,
        fold_dir=fold_dir,
        fold_name=f"bert_fold_{fold_idx}",
    )

    fold_best_paths.append(best_path)

    # -----------------------------------------------------------
    # Reload best model for evaluation on its validation set
    # -----------------------------------------------------------
    print(f"\nReloading BEST MODEL for fold {fold_idx}: {best_path}\n")

    best_model = create_model()
    best_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    best_model.to(DEVICE)
    best_model.eval()

    eval_args = TrainingArguments(
        output_dir=os.path.join(fold_dir, "eval_logs"),
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        fp16=True,
        report_to="none",
    )

    eval_trainer = FocalTrainer(
        class_weights=class_weights_tensor,
        gamma=0.5,
        smoothing=0.05,
        model=best_model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=val_fold,
        compute_metrics=compute_metrics,
    )

    val_metrics = eval_trainer.evaluate(val_fold)
    fold_val_metrics.append(val_metrics)

    print(f"Fold {fold_idx} Validation Accuracy: {val_metrics['eval_accuracy']:.4f}")
    print(f"Fold {fold_idx} Validation F1 Score: {val_metrics['eval_f1']:.4f}\n")

# ---------------------------------------------------------------
# SUMMARY ACROSS ALL 3 FOLDS
# ---------------------------------------------------------------
print("\n========== CROSS-VALIDATION SUMMARY ==========\n")
for i, (path, metrics) in enumerate(zip(fold_best_paths, fold_val_metrics), start=1):
    print(
        f"Fold {i} — Best ckpt: {path} | "
        f"Val Acc: {metrics['eval_accuracy']:.4f} | "
        f"Val F1: {metrics['eval_f1']:.4f}"
    )

print("\nAll folds completed successfully.\n")


# ================================================================
# PART 4 — FINAL ENSEMBLE EVALUATION ON TEST SET
# ================================================================

print("\n========== FINAL ENSEMBLE EVALUATION ==========\n")

test_ds_tok = tokenized_dataset["test"]

test_loader = DataLoader(
    test_ds_tok,
    batch_size=32,
    shuffle=False,
)

# ---------------------------------------------------------------
# 1. Load best model from each fold
# ---------------------------------------------------------------
ensemble_models = []

print("Loading best models from each fold...\n")
for i, best_path in enumerate(fold_best_paths, start=1):
    print(f"Fold {i}: loading {best_path}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,           # "google/bert_uncased_L-4_H-512_A-8"
        num_labels=num_classes,
    )
    model.gradient_checkpointing_enable()
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    ensemble_models.append(model)

print("\nAll fold models loaded.\n")

# ---------------------------------------------------------------
# 2. Ensemble inference (logits sum)
# ---------------------------------------------------------------
all_logits = []
all_labels = []

print("Running ensemble inference on test set...\n")

with torch.no_grad():
    for batch in test_loader:
        labels_np = batch["label"].cpu().numpy()
        all_labels.append(labels_np)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        summed_logits = None

        for model in ensemble_models:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            if summed_logits is None:
                summed_logits = logits
            else:
                summed_logits += logits

        all_logits.append(summed_logits.cpu())

all_logits = torch.cat(all_logits, dim=0).numpy()
all_labels = np.concatenate(all_labels, axis=0)

# ---------------------------------------------------------------
# 3. Final metrics
# ---------------------------------------------------------------
final_metrics = compute_metrics((all_logits, all_labels))

print("\n========== FINAL TEST METRICS (ENSEMBLE) ==========\n")
print(f"Accuracy : {final_metrics['accuracy']:.4f}")
print(f"Precision: {final_metrics['precision']:.4f}")
print(f"Recall   : {final_metrics['recall']:.4f}")
print(f"F1 Score : {final_metrics['f1']:.4f}\n")

# ---------------------------------------------------------------
# 4. Classification report
# ---------------------------------------------------------------
print("========== CLASSIFICATION REPORT ==========\n")
print(
    classification_report(
        all_labels,
        all_logits.argmax(axis=-1),
        digits=4,
        zero_division=0,
    )
)

# ---------------------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------------------
print("\n========== CONFUSION MATRIX ==========\n")
cm = confusion_matrix(all_labels, all_logits.argmax(axis=-1))
print("Shape:", cm.shape)
print(cm)

print("\nDone.\n")