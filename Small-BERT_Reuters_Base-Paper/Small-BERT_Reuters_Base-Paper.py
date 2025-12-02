import os
import json
import pickle
import time
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file

# =========================================================
# Device: force GPU (CUDA) – do not allow CPU fallback
# =========================================================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available. Please enable a GPU before running this script.")
DEVICE = torch.device("cuda")

# =========================================================
# 1. Load Reuters from local .npz (same as tf.keras.datasets.reuters)
# =========================================================
NUM_WORDS = 10000

home = os.path.expanduser("~")
data_dir = os.path.join(home, ".keras", "datasets")

npz_path = os.path.join(data_dir, "reuters.npz")
word_index_json = os.path.join(data_dir, "reuters_word_index.json")
word_index_pkl = os.path.join(data_dir, "reuters_word_index.pkl")

print(f"Loading Reuters from: {npz_path}")
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"Could not find {npz_path}")

with np.load(npz_path, allow_pickle=True) as f:
    xs, labels = f["x"], f["y"]

# replicate tf.keras.reuters.load_data logic (no maxlen, skip_top=0)
seed = 113
test_split = 0.2
start_char = 1
oov_char = 2
index_from = 3
skip_top = 0

rng = np.random.RandomState(seed)
indices = np.arange(len(xs))
rng.shuffle(indices)
xs = xs[indices]
labels = labels[indices]

# add start token + shift indices
xs = [[start_char] + [w + index_from for w in x] for x in xs]

# apply num_words cutoff (OOV handling)
xs = [
    [w if skip_top <= w < NUM_WORDS else oov_char for w in x]
    for x in xs
]

idx = int(len(xs) * (1 - test_split))
x_train, y_train = np.array(xs[:idx], dtype="object"), np.array(labels[:idx])
x_test, y_test = np.array(xs[idx:], dtype="object"), np.array(labels[idx:])

print(f"Train samples: {len(x_train)}")
print(f"Test samples:  {len(x_test)}")
print(f"Total samples: {len(x_train) + len(x_test)}")

num_classes = int(np.max(y_train) + 1)
print(f"Num classes: {num_classes}")

# =========================================================
# 2. Load word_index (download if missing) + decode to text
# =========================================================
if os.path.exists(word_index_json):
    print("Loading word_index from JSON...")
    with open(word_index_json, "r", encoding="utf-8") as f:
        word_index = json.load(f)
elif os.path.exists(word_index_pkl):
    print("Loading word_index from PKL...")
    with open(word_index_pkl, "rb") as f:
        word_index = pickle.load(f)
else:
    import urllib.request

    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters_word_index.json"
    print(f"word_index not found locally, downloading from {url} ...")
    urllib.request.urlretrieve(url, word_index_json)

    with open(word_index_json, "r", encoding="utf-8") as f:
        word_index = json.load(f)

# Keras reserves 0–3 for special tokens, so we shift by +3
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "[PAD]"
reverse_word_index[1] = "[START]"
reverse_word_index[2] = "[UNK]"
reverse_word_index[3] = "[UNUSED]"


def decode_review(sequence):
    return " ".join(reverse_word_index.get(i, "[UNK]") for i in sequence)


train_texts_raw = [decode_review(seq) for seq in x_train]
test_texts_raw = [decode_review(seq) for seq in x_test]

train_labels = y_train.tolist()
test_labels = y_test.tolist()

print("\nExample decoded text:")
print(train_texts_raw[0][:300], "...")
print("Label:", train_labels[0])

print(f"\nUsing full Keras train split for training/validation: {len(train_texts_raw)} samples")
print(f"Held-out test set: {len(test_texts_raw)} samples")

# =========================================================
# 3. Build HF Dataset (train + test; we'll create splits later)
# =========================================================
print("\nCreating HuggingFace Dataset...")

train_ds = Dataset.from_dict({"text": train_texts_raw, "label": train_labels})
test_ds = Dataset.from_dict({"text": test_texts_raw, "label": test_labels})

dataset = DatasetDict(
    {
        "train": train_ds,
        "test": test_ds,
    }
)

print(dataset)

# =========================================================
# 4. Tokenizer and tokenization (Small BERT-compatible)
# =========================================================
# Small BERT (4-layer, H=512, A=8)
model_name = "google/bert_uncased_L-4_H-512_A-8"

print(f"\nLoading tokenizer: bert-base-uncased (vocab compatible with Small BERT)")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

MAX_LEN = 300  # as per plan (reduce to 256 only if OOM)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )


print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_batch, batched=True)

# We no longer need the raw text for training
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

print(tokenized_dataset)

# =========================================================
# 5. Compute class weights for imbalanced loss (on full train)
# =========================================================
print("\nComputing class weights on full training set...")
counter = Counter(train_labels)
class_counts = np.zeros(num_classes, dtype=np.float32)
for lbl, cnt in counter.items():
    class_counts[int(lbl)] = cnt

# Inverse-frequency weighting
class_weights = 1.0 / np.maximum(class_counts, 1.0)
class_weights = class_weights * (num_classes / class_weights.sum())  # normalize a bit
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print("Class weights:", class_weights_tensor)

# =========================================================
# 6. Custom output class for model (needed for Trainer compatibility)
# =========================================================
class ModelOutput:
    def __init__(self, logits):
        self.logits = logits
    
    def __getitem__(self, idx):
        # Support subscripting for Trainer compatibility
        if isinstance(idx, slice):
            # Handle slice notation like outputs[1:]
            # Return tuple of (logits,) for outputs[1:]
            return (self.logits,)
        elif idx == 0:
            return None  # loss (we compute it in custom trainer)
        elif idx == 1:
            return self.logits
        else:
            raise IndexError(f"Index {idx} out of range")

# =========================================================
# 7. Custom Small BERT classifier head (512 -> 256 -> Dropout -> 46)
# =========================================================
class SmallBertReutersClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int, dropout_prob: float = 0.3, intermediate_dim: int = 256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name, use_safetensors=True)
        hidden_size = self.bert.config.hidden_size  # 512 for Small BERT

        self.dropout = nn.Dropout(dropout_prob)
        self.intermediate = nn.Linear(hidden_size, intermediate_dim)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(intermediate_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Remove 'labels' from kwargs if present (Trainer passes it during eval but BERT doesn't accept it)
        kwargs.pop('labels', None)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        # pooled_output is outputs[1] or outputs.pooler_output
        pooled_output = outputs[1] if len(outputs) > 1 else outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.intermediate(x)
        x = self.activation(x)
        x = self.dropout2(x)
        logits = self.classifier(x)
        # Return proper ModelOutput object
        return ModelOutput(logits=logits)


# =========================================================
# 7. Metrics function (accuracy + weighted precision/recall/F1)
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =========================================================
# 8. Custom Trainer with Focal Loss + class weights + label smoothing
# =========================================================
class WeightedFocalTrainer(Trainer):
    def __init__(self, class_weights, gamma=2.0, label_smoothing=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self._class_weights_on_device = None

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int = None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self._class_weights_on_device is None:
            self._class_weights_on_device = self.class_weights.to(logits.device)

        # Multi-class focal loss with label smoothing & class weights
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        probs = log_probs.exp()
        num_classes = logits.size(-1)

        # Label-smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / num_classes)
            true_dist.scatter_(
                1,
                labels.unsqueeze(1),
                1.0 - self.label_smoothing + self.label_smoothing / num_classes,
            )

        alpha = self._class_weights_on_device.unsqueeze(0)  # (1, C) -> broadcast to (B, C)
        focal_factor = (1.0 - probs) ** self.gamma

        loss = - (true_dist * alpha * focal_factor * log_probs).sum(dim=-1)  # (B,)
        loss = loss.mean()

        return (loss, outputs) if return_outputs else loss


# =========================================================
# 9. Callback: epoch metrics + early stopping + timing + best model save
# =========================================================
class EpochMetricsAndEarlyStoppingCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, best_model_dir, patience=5, min_delta=1e-4):
        super().__init__()
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.best_model_dir = best_model_dir
        self.patience = patience
        self.min_delta = min_delta

        self.best_metric = None
        self.best_epoch = None
        self.patience_counter = 0

        self.train_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        print(f"\nTraining on device: {self.trainer.args.device}\n")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        # Evaluate on validation set
        metrics = self.trainer.evaluate(self.eval_dataset)
        epoch = state.epoch
        epoch_time = time.time() - self.epoch_start_time

        val_loss = metrics.get("eval_loss")
        val_acc = metrics.get("eval_accuracy")
        val_f1 = metrics.get("eval_f1")

        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (float, int)) else "NA"

        print(
            f"[Epoch {int(epoch)}/{int(args.num_train_epochs)}] "
            f"val_loss={fmt(val_loss)} "
            f"val_acc={fmt(val_acc)} "
            f"val_f1={fmt(val_f1)} "
            f"time={epoch_time:.1f}s"
        )

        # Early stopping + best model saving (based on val_f1)
        if val_f1 is not None:
            if (self.best_metric is None) or (val_f1 - self.best_metric > self.min_delta):
                self.best_metric = val_f1
                self.best_epoch = int(epoch)
                self.patience_counter = 0
                print(
                    f"  -> New best model (val_f1={val_f1:.4f}) at epoch {int(epoch)}, "
                    f"saving to {self.best_model_dir}"
                )
                os.makedirs(self.best_model_dir, exist_ok=True)
                self.trainer.save_model(self.best_model_dir)
            else:
                self.patience_counter += 1
                print(
                    f"  -> No improvement in val_f1 for {self.patience_counter} epoch(s) "
                    f"(patience={self.patience})."
                )

                if self.patience_counter >= self.patience:
                    print("  -> Early stopping triggered.")
                    control.should_training_stop = True

        return control

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time if self.train_start_time else 0.0
        if self.best_metric is not None:
            print(
                f"\nTraining finished in {total_time/60:.1f} min. "
                f"Best val_f1={self.best_metric:.4f} at epoch {self.best_epoch}."
            )
        else:
            print(f"\nTraining finished in {total_time/60:.1f} min (no validation metric recorded).")


# =========================================================
# 10. Helper: train one run (used for LR search & CV folds)
# =========================================================
def train_one_run(
    train_dataset,
    val_dataset,
    learning_rate,
    output_dir,
    run_name,
    class_weights_tensor,
    num_epochs=10,
    warmup_ratio=0.15,
):
    print(f"\n========== Training run: {run_name} (lr={learning_rate}) ==========")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,                   # early stopping will cut this
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,                 # effective batch size 32
        learning_rate=learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,                             # gradient clipping for stability
        logging_strategy="no",
        eval_strategy="epoch",                         # renamed from evaluation_strategy
        save_strategy="no",                            # we manage best-model saving ourselves
        fp16=True,
        report_to="none",
        seed=42,
        disable_tqdm=False,
    )

    model = SmallBertReutersClassifier(
        base_model_name=model_name,
        num_labels=num_classes,
        dropout_prob=0.3,
        intermediate_dim=256,
    ).to(DEVICE)

    trainer = WeightedFocalTrainer(
        class_weights=class_weights_tensor,
        gamma=2.0,
        label_smoothing=0.1,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    best_model_dir = os.path.join(output_dir, "best")

    callback = EpochMetricsAndEarlyStoppingCallback(
        trainer=trainer,
        eval_dataset=val_dataset,
        best_model_dir=best_model_dir,
        patience=5,
        min_delta=1e-4,
    )
    trainer.add_callback(callback)

    trainer.train()

    best_val_f1 = callback.best_metric
    best_epoch = callback.best_epoch

    print(
        f"\nRun '{run_name}' finished. Best val_f1={best_val_f1:.4f} at epoch {best_epoch}.\n"
    )

    return best_val_f1, best_epoch, best_model_dir


# =========================================================
# 11. Phase 6 – Learning rate search (single stratified split)
# =========================================================
print("\n========== Phase 6: Learning Rate Search ==========")

N_train = len(train_labels)
indices = np.arange(N_train)

train_idx_lr, val_idx_lr = train_test_split(
    indices,
    test_size=0.1,
    random_state=42,
    stratify=train_labels,
)

train_idx_lr = train_idx_lr.tolist()
val_idx_lr = val_idx_lr.tolist()

tokenized_train_lr = tokenized_dataset["train"].select(train_idx_lr)
tokenized_val_lr = tokenized_dataset["train"].select(val_idx_lr)

lr_candidates = [1e-5, 2e-5, 3e-5, 5e-5]
lr_results = {}

base_output_dir = "reuters-small-bert-max"
os.makedirs(base_output_dir, exist_ok=True)

for lr in lr_candidates:
    lr_tag = str(lr).replace(".", "p").replace("-", "m")  # simple safe tag
    run_name = f"lr_{lr_tag}"
    output_dir = os.path.join(base_output_dir, run_name)

    best_val_f1, best_epoch, best_model_dir = train_one_run(
        train_dataset=tokenized_train_lr,
        val_dataset=tokenized_val_lr,
        learning_rate=lr,
        output_dir=output_dir,
        run_name=run_name,
        class_weights_tensor=class_weights_tensor,
        num_epochs=10,
        warmup_ratio=0.15,
    )

    lr_results[lr] = {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "best_model_dir": best_model_dir,
    }

print("\nLR search results:")
for lr, info in lr_results.items():
    print(f"  lr={lr}: best_val_f1={info['best_val_f1']:.4f} at epoch {info['best_epoch']}")

# Select best LR
best_lr = max(lr_results.items(), key=lambda x: x[1]["best_val_f1"])[0]
print(f"\nSelected best learning rate for subsequent training: {best_lr}\n")

# =========================================================
# 12. Phase 7 – 3-fold stratified cross-validation training
# =========================================================
print("========== Phase 7: 3-Fold Stratified Cross-Validation ==========")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
labels_array = np.array(train_labels)

fold_best_dirs = []
fold_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels_array), start=1):
    print(f"\n===== Fold {fold_idx} / 3 =====")

    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()

    tokenized_train_fold = tokenized_dataset["train"].select(train_idx)
    tokenized_val_fold = tokenized_dataset["train"].select(val_idx)

    fold_run_name = f"cv_fold_{fold_idx}"
    fold_output_dir = os.path.join(base_output_dir, fold_run_name)

    best_val_f1, best_epoch, best_model_dir = train_one_run(
        train_dataset=tokenized_train_fold,
        val_dataset=tokenized_val_fold,
        learning_rate=best_lr,
        output_dir=fold_output_dir,
        run_name=fold_run_name,
        class_weights_tensor=class_weights_tensor,
        num_epochs=10,
        warmup_ratio=0.15,
    )

    # Reload best model for this fold and evaluate on its validation set
    print(f"Reloading best model for fold {fold_idx} to evaluate on its validation set...")
    # Load the saved model using HuggingFace's save/load mechanism
    # The Trainer saves the entire model, so we need to load just the BERT part and rebuild our classifier
    fold_model = SmallBertReutersClassifier(
        base_model_name=model_name,
        num_labels=num_classes,
        dropout_prob=0.3,
        intermediate_dim=256,
    ).to(DEVICE)
    # Load the saved state dict from the checkpoint
    checkpoint_path = os.path.join(best_model_dir, "model.safetensors")
    if os.path.exists(checkpoint_path):
        state_dict = load_file(checkpoint_path)
        fold_model.load_state_dict(state_dict)
    else:
        print(f"Warning: Could not find {checkpoint_path}, using freshly initialized model")

    # Build a temporary trainer just for evaluation
    eval_trainer = WeightedFocalTrainer(
        class_weights=class_weights_tensor,
        gamma=2.0,
        label_smoothing=0.1,
        model=fold_model,
        args=TrainingArguments(
            output_dir=os.path.join(fold_output_dir, "eval"),
            per_device_eval_batch_size=32,
            do_train=False,
            do_eval=True,
            fp16=True,
            report_to="none",
        ),
        train_dataset=None,
        eval_dataset=tokenized_val_fold,
        compute_metrics=compute_metrics,
    )

    val_metrics = eval_trainer.evaluate(tokenized_val_fold)
    print(f"Fold {fold_idx} validation metrics:", val_metrics)

    fold_best_dirs.append(best_model_dir)
    fold_metrics.append(val_metrics)

print("\nCross-validation fold summary:")
for i, (d, m) in enumerate(zip(fold_best_dirs, fold_metrics), start=1):
    print(f"  Fold {i}: best_dir={d}, val_f1={m['eval_f1']:.4f}, val_acc={m['eval_accuracy']:.4f}")

# =========================================================
# 13. Phase 8 – Ensemble on held-out test set
# =========================================================
print("\n========== Phase 8: Ensemble Evaluation on Test Set ==========")

# Load all fold best models
ensemble_models = []
for i, best_dir in enumerate(fold_best_dirs, start=1):
    checkpoint_path = os.path.join(best_dir, "model.safetensors")
    print(f"Loading fold {i} model from {checkpoint_path}")
    m = SmallBertReutersClassifier(
        base_model_name=model_name,
        num_labels=num_classes,
        dropout_prob=0.3,
        intermediate_dim=256,
    ).to(DEVICE)
    if os.path.exists(checkpoint_path):
        state_dict = load_file(checkpoint_path)
        m.load_state_dict(state_dict)
    else:
        print(f"Warning: Could not find {checkpoint_path}, using freshly initialized model")
    m.eval()
    ensemble_models.append(m)

test_ds_tok = tokenized_dataset["test"]
test_loader = DataLoader(test_ds_tok, batch_size=32, shuffle=False)

all_logits = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        labels_batch = batch["label"].cpu().numpy()
        all_labels.append(labels_batch)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        logits_sum = None
        for m in ensemble_models:
            outputs = m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            if logits_sum is None:
                logits_sum = outputs.logits
            else:
                logits_sum = logits_sum + outputs.logits

        all_logits.append(logits_sum.cpu())

all_logits = torch.cat(all_logits, dim=0).numpy()
all_labels = np.concatenate(all_labels, axis=0)

ensemble_metrics = compute_metrics((all_logits, all_labels))
print("\nEnsemble test metrics (held-out test set):", ensemble_metrics)

# =========================================================
# 14. Phase 9 – Error analysis (per-class F1 + confusion matrix)
# =========================================================
print("\n========== Phase 9: Error Analysis ==========")

test_preds = all_logits.argmax(axis=-1)

print("\nClassification report (per-class F1):")
print(classification_report(all_labels, test_preds, digits=4, zero_division=0))

print("Confusion matrix (shape: {}):".format(confusion_matrix(all_labels, test_preds).shape))
print(confusion_matrix(all_labels, test_preds))

print("\nDone.")
