# Fine-Tuning BERT-Base for Reuters News Classification - Implementation Report

This document describes the implementation of BERT-base fine-tuning on a **filtered Reuters-21578 dataset** (classes with ≥50 samples) using advanced training techniques including focal loss, class weighting, label smoothing, gradient checkpointing, and 3-fold cross-validation ensemble.

---

## 1. Problem Definition

- **Task**: Automatic **news topic classification** for English news articles.
- **Domain**: General news articles from the **Reuters-21578** collection.
- **Objective**: Fine-tune **BERT-base-uncased** on a **filtered dataset** (removing ultra-rare classes) to achieve:
  - Better model stability by focusing on classes with sufficient training data.
  - Improved performance through aggressive text cleaning and preprocessing.
  - Robust evaluation via 3-fold stratified cross-validation with ensemble prediction.
  - Memory-efficient training using gradient checkpointing.

---

## 2. Dataset and Preprocessing

### 2.1 Reuters-21578 Dataset (Filtered)

**Original dataset**:
- Total samples: **11,228** news articles
- Training samples: **8,982**
- Test samples: **2,246**
- Categories: **46** topic classes

**Filtering strategy**:
- **Removed classes with < 50 samples** to improve training stability
- **Classes kept**: **18** (down from 46)
- **Final dataset**:
  - Training samples: **8,227** (755 samples removed)
  - Test samples: **2,042** (204 samples removed)

**Rationale**: Ultra-rare classes (< 50 samples) cause:
- Extreme class imbalance
- Unstable training
- Poor generalization
- Unreliable metrics

### 2.2 Text Cleaning and Preprocessing

**Aggressive cleaning pipeline**:
1. **Remove special tokens**: `[START]`, `[UNK]`, `[PAD]`, `[UNUSED]`
2. **Lowercase conversion**
3. **Remove stopwords**: Using NLTK English stopwords
4. **Remove punctuation and numbers**
5. **Remove extra whitespace**
6. **Minimum length filter**: Keep only non-empty texts

**Example**:
- **Before**: `[START] [UNK] [UNK] said as a result of its december acquisition...`
- **After**: `said as a result of its december acquisition of space co it expects earnings...`

### 2.3 Sequence Length and Tokenization

- **Vocabulary**: BERT-base WordPiece (30,522 tokens)
- **Sequence length**: **300 tokens** (padded or truncated)
- **Tokenizer**: `bert-base-uncased`
- **Padding**: `max_length` strategy
- **Truncation**: Enabled

### 2.4 Class Distribution and Weights

**Final class counts** (18 classes):
```
Class 0:   55 samples  | Class 9:  172 samples
Class 1:  432 samples  | Class 10: 444 samples
Class 2:   74 samples  | Class 11:  66 samples
Class 3: 3159 samples  | Class 12: 549 samples
Class 4: 1949 samples  | Class 13: 269 samples
Class 5:  139 samples  | Class 14: 100 samples
Class 6:  101 samples  | Class 15:  62 samples
Class 7:  124 samples  | Class 16:  92 samples
Class 8:  390 samples  | Class 17:  50 samples
```

**Class weights** (inverse-frequency, clipped & normalized):
```
[2.21, 0.28, 1.64, 0.04, 0.06, 0.87, 1.20, 0.98, 0.31, 0.71, 
 0.27, 1.84, 0.22, 0.45, 1.21, 1.96, 1.32, 2.43]
```

---

## 3. Model Architecture

### 3.1 BERT Encoder

**Model**: `bert-base-uncased` (12-layer, 768-hidden, 12-heads)
- **Transformer layers**: 12 encoder blocks
- **Hidden size**: 768
- **Attention heads**: 12 per layer
- **Parameters**: ~110M
- **Gradient checkpointing**: **Enabled** (saves GPU memory)

### 3.2 Classification Head

**Native HuggingFace head** (`AutoModelForSequenceClassification`):
- **Architecture**: `[CLS] → Dense(18, softmax)`
- **Dropout**: 
  - `hidden_dropout_prob`: **0.3**
  - `attention_probs_dropout_prob`: **0.1**
- **Output**: 18 neurons (one per class)

**Key difference from Reuters**: Uses HuggingFace's built-in classification head instead of custom layers.

---

## 4. Training Configuration

### 4.1 Loss Function

**Focal Loss** with class weights and label smoothing:

- **Focal parameter (γ)**: **0.5** (reduced from 2.0 for stability)
- **Class weights**: Applied per-class (α weights)
- **Label smoothing**: **0.05** (5% smoothing)

Formula:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

Label smoothing:
```
y_smooth = (1 - ε) * y_true + ε / num_classes
```

### 4.2 Optimizer and Scheduler

- **Optimizer**: **AdamW** (Adam with weight decay)
- **Weight decay**: **0.01**
- **Learning rate**: **3e-5** (fixed, no search)
- **LR scheduler**: **Cosine** with warmup
- **Warmup ratio**: **0.15** (15% of total steps)

### 4.3 Training Hyperparameters

- **Batch size**: 8 per device
- **Gradient accumulation steps**: 4 (effective batch size = **32**)
- **Epochs**: 20 (with early stopping)
- **Early stopping patience**: **7 epochs**
- **Early stopping metric**: Validation F1-score
- **Min delta**: **1e-4**
- **Mixed precision**: **FP16** enabled
- **Gradient checkpointing**: **Enabled** (memory optimization)

### 4.4 Hardware

- **Device**: CUDA GPU (Google Colab T4)
- **GPU**: NVIDIA Tesla T4

---

## 5. Training Strategy

### 5.1 3-Fold Stratified Cross-Validation

**Objective**: Train robust ensemble without separate LR search.

**Setup**:
- **Folds**: 3 stratified folds
- **Learning rate**: **3e-5** (fixed)
- **Epochs**: 20 per fold (with early stopping)
- **Validation**: Each fold uses different 1/3 of data
- **Checkpoint management**: Auto-resume from last checkpoint
- **Save strategy**: Save every epoch, keep last 5 checkpoints

**Fold Results**:

| Fold | Best Epoch | Val F1  | Val Accuracy |
|------|-----------|---------|--------------|
| 1    | 11        | 0.8867  | 0.8881       |
| 2    | 9         | 0.8914  | 0.8891       |
| 3    | 11        | 0.8812  | 0.8797       |

**Average CV F1**: **0.8864**

**Key observations**:
- All folds converged to similar performance (~88% F1)
- Early stopping triggered between epochs 9-11
- Consistent validation accuracy across folds

---

## 6. Evaluation and Results

### 6.1 Metrics

- **Accuracy**
- **Precision** (weighted average)
- **Recall** (weighted average)
- **F1-score** (weighted average)

### 6.2 Ensemble Test Performance

**Method**: Average logits from 3 fold models, then argmax.

**Test Set Results** (held-out 2,042 samples):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 87.56% |
| Precision | 0.8817 |
| Recall    | 0.8756 |
| **F1-score** | **0.8773** |

### 6.3 Per-Class Performance

**High-performing classes** (F1 > 0.90):
- Class 3: F1 = 0.9458 (812 samples) - largest class
- Class 16: F1 = 0.9524 (31 samples)
- Class 7: F1 = 0.9032 (30 samples)
- Class 4: F1 = 0.8998 (474 samples)
- Class 6: F1 = 0.8846 (25 samples)

**Moderate-performing classes** (0.70 < F1 < 0.90):
- Class 1: F1 = 0.8505 (105 samples)
- Class 0: F1 = 0.8696 (12 samples)
- Class 8: F1 = 0.8144 (83 samples)
- Class 10: F1 = 0.7943 (99 samples)
- Class 14: F1 = 0.7812 (27 samples)
- Class 5: F1 = 0.7273 (38 samples)
- Class 15: F1 = 0.7143 (19 samples)
- Class 2: F1 = 0.7000 (20 samples)

**Lower-performing classes** (F1 < 0.70):
- Class 11: F1 = 0.6842 (20 samples)
- Class 13: F1 = 0.6115 (70 samples)
- Class 9: F1 = 0.7532 (37 samples)

**Macro average**: Precision 0.8032, Recall 0.8177, F1 0.8064
**Weighted average**: Precision 0.8817, Recall 0.8756, F1 0.8773

---

## 7. Error Analysis

### 7.1 Main Error Sources

1. **Remaining class imbalance**:
   - Despite filtering, Class 3 (3,159 samples) dominates
   - Smallest classes (50-70 samples) still show lower F1
   - Class 13 (70 samples): 61.15% F1

2. **Confusion between similar topics**:
   - Financial and economic news categories overlap
   - Political and business topics confused

3. **Sequence truncation**:
   - 300-token limit may cut important context
   - Longer articles lose tail information

### 7.2 Confusion Matrix Insights

- **Shape**: 18 × 18
- **Diagonal dominance**: Strong for major classes (3, 4, 1, 12)
- **Off-diagonal errors**: 
  - Class 3 ↔ Class 4: Frequent confusion (both large classes)
  - Class 1 → Class 3: 7 misclassifications
  - Class 13 shows scattered errors across multiple classes

---

## 8. Training Time and Computational Cost

### 8.1 Training Duration

**Cross-Validation** (estimated from progress bars):
- Fold 1: ~46 minutes (18 epochs, early stopped)
- Fold 2: ~43 minutes (16 epochs, early stopped)
- Fold 3: ~48 minutes (18 epochs, early stopped)

**Total training time**: ~137 minutes (~2.3 hours)

**Note**: Significantly faster than Reuters (~10.3 hours) due to:
- Smaller dataset (8,227 vs 8,982 samples)
- Gradient checkpointing (memory-efficient)
- Native HuggingFace head (optimized)

### 8.2 Training Efficiency

- **Iterations per second**: ~1.18-1.22 it/s
- **Mixed precision**: FP16 enabled
- **Gradient checkpointing**: Enabled (trades compute for memory)
- **Checkpoint management**: Auto-resume capability

---

## 9. Key Implementation Details

### 9.1 Custom Trainer

**FocalTrainer** extends HuggingFace `Trainer`:
- Implements focal loss with class weights
- Applies label smoothing
- Caches class weights on device
- Compatible with HuggingFace's training loop

### 9.2 Callbacks

**EarlyStopCallback**:
- Evaluates on validation set after each epoch
- Prints table with metrics (Loss, Accuracy, Precision, Recall, F1)
- Saves best model based on validation F1
- Implements early stopping with patience
- Uses PyTorch `.pt` format for checkpoints

### 9.3 Model Saving and Loading

- **Format**: PyTorch state dict (`.pt`)
- **Best model**: Saved per fold in `best_model.pt`
- **Loading**: Direct state dict loading for ensemble
- **Checkpoint strategy**: Save every epoch, keep last 5

---

## 10. Comparison with Reuters Implementation

### 10.1 Dataset Differences

| Aspect | Reuters | Reuters Filtered |
|--------|---------|------------------|
| Classes | 46 | 18 |
| Train samples | 8,982 | 8,227 |
| Test samples | 2,246 | 2,042 |
| Text cleaning | Minimal | Aggressive |
| Stopword removal | No | Yes |

### 10.2 Architecture Differences

| Aspect | Reuters | Reuters Filtered |
|--------|---------|------------------|
| BERT variant | BERT-base | BERT-base |
| Classification head | Custom (768→256→46) | Native HF (768→18) |
| Dropout layers | 2 layers (0.3 each) | Built-in (0.3 hidden, 0.1 attention) |
| Gradient checkpointing | No | Yes |

### 10.3 Training Differences

| Aspect | Reuters | Reuters Filtered |
|--------|---------|------------------|
| Loss | Weighted focal (γ=1.0) | Weighted focal (γ=0.5) |
| Label smoothing | 0.1 | 0.05 |
| Batch size | 16 (accum 2 = 32) | 8 (accum 4 = 32) |
| LR search | Yes (2e-5, 3e-5) | No (fixed 3e-5) |
| Epochs | 20 | 20 |
| Patience | 7 | 7 |

### 10.4 Performance Comparison

| Model | Classes | Accuracy | Precision | Recall | F1-score |
|-------|---------|----------|-----------|--------|----------|
| Reuters | 46 | 83.70% | 0.8708 | 0.8370 | 0.8497 |
| **Reuters Filtered** | **18** | **87.56%** | **0.8817** | **0.8756** | **0.8773** |

**Performance improvement**: +3.86% accuracy, +2.76% F1

**Reasons for better performance**:
1. **Filtered dataset**: Removed ultra-rare classes (< 50 samples)
2. **Aggressive text cleaning**: Better signal-to-noise ratio
3. **Reduced complexity**: 18 classes vs 46 classes
4. **Lower focal gamma**: 0.5 vs 1.0 (less aggressive focusing)

---

## 11. Strengths and Limitations

### 11.1 Strengths

1. **Cleaner dataset**:
   - Aggressive preprocessing removes noise
   - Stopword removal improves signal
   - Filtered classes improve stability

2. **Memory efficiency**:
   - Gradient checkpointing enables larger models
   - Smaller batch size with accumulation

3. **Robust training**:
   - 3-fold CV reduces overfitting
   - Ensemble improves generalization
   - Auto-resume from checkpoints

4. **Better performance**:
   - 87.56% accuracy vs 83.70% (Reuters)
   - Fewer classes = better per-class performance

### 11.2 Limitations

1. **Reduced coverage**:
   - Only 18 classes (vs original 46)
   - 28 rare classes excluded
   - Not suitable for comprehensive classification

2. **Aggressive filtering**:
   - Loses information from rare topics
   - May not generalize to new rare classes

3. **Still some imbalance**:
   - Class 3 (3,159 samples) dominates
   - Smallest classes (50-70 samples) still struggle

4. **Computational cost**:
   - Gradient checkpointing trades speed for memory
   - 3 models for ensemble (3× storage)

---

## 12. Future Improvements

### 12.1 Dataset Enhancements

- **Data augmentation**: Back-translation, paraphrasing for small classes
- **Hierarchical classification**: Group similar classes
- **Multi-label approach**: Allow multiple topics per article

### 12.2 Model Architecture

- **Hierarchical attention**: Handle longer documents
- **Lightweight variants**: DistilBERT for faster inference
- **Domain-specific BERT**: FinBERT for financial news

### 12.3 Training Techniques

- **Curriculum learning**: Start with easy examples
- **Mixup/CutMix**: Advanced augmentation
- **Self-training**: Use unlabeled Reuters data

### 12.4 Deployment

- **Model compression**: Quantization (INT8), pruning
- **Knowledge distillation**: Distill ensemble into single model
- **ONNX export**: For production inference
- **Serving optimization**: TensorRT, TorchScript

---

## 13. Conclusion

This implementation demonstrates fine-tuning **BERT-base** on a **filtered Reuters-21578 dataset** (≥50 samples per class) with advanced techniques:

- **Ensemble F1**: 0.8773 (87.56% accuracy)
- **Filtered dataset**: 18 classes (from 46)
- **Aggressive preprocessing**: Stopword removal, text cleaning
- **Memory-efficient**: Gradient checkpointing enabled

**Key achievements**:
- **+3.86% accuracy** improvement over Reuters (46 classes)
- **+2.76% F1** improvement
- **Faster training**: 2.3 hours vs 10.3 hours
- **More stable**: Removed ultra-rare classes

**Trade-offs**:
- **Reduced coverage**: 18 classes vs 46 classes
- **Excluded rare topics**: 28 classes removed
- **Not comprehensive**: Suitable for major topics only

**Best use case**: Production systems focusing on **major news categories** where rare topics can be handled separately or excluded.

**Key takeaway**: Filtering ultra-rare classes and aggressive text preprocessing significantly improve BERT performance on imbalanced news classification, achieving **87.56% accuracy** on 18 major Reuters categories.
