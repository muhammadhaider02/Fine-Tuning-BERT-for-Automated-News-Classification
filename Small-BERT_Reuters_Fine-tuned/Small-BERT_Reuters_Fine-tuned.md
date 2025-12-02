# Fine-Tuning Small-BERT for Reuters News Classification - Implementation Report

This document describes the implementation of Small-BERT fine-tuning on a **filtered Reuters-21578 dataset** (classes with ≥50 samples) using advanced training techniques including focal loss, class weighting, label smoothing, gradient checkpointing, and 3-fold cross-validation ensemble.

---

## 1. Problem Definition

- **Task**: Automatic **news topic classification** for English news articles.
- **Domain**: General news articles from the **Reuters-21578** collection.
- **Objective**: Fine-tune **Small BERT (L-4_H-512_A-8)** on a **filtered dataset** (removing ultra-rare classes) to achieve:
  - Better model stability by focusing on classes with sufficient training data.
  - Improved performance through minimal text cleaning and preprocessing.
  - Robust evaluation via 3-fold stratified cross-validation with ensemble prediction.
  - Memory-efficient training using gradient checkpointing.
  - Faster training and inference compared to BERT-base while maintaining competitive performance.

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

**Minimal cleaning pipeline** (similar to AG News approach):
1. **Lowercase conversion**
2. **Remove Keras artifact tokens**: `[START]`, `[UNK]`, `[PAD]`, `[UNUSED]`
3. **Collapse extra whitespace**
4. **Minimum length filter**: Drop documents with < 3 tokens

**Key difference from BERT-Base Reuters**: 
- **No stopword removal**
- **No punctuation removal**
- **No number removal**
- Preserves more contextual information

**Example**:
- **Before**: `[START] [UNK] [UNK] said as a result of its december acquisition...`
- **After**: `said as a result of its december acquisition of space co it expects earnings...`

### 2.3 Sequence Length and Tokenization

- **Model**: `google/bert_uncased_L-4_H-512_A-8` (Small BERT)
- **Vocabulary**: BERT WordPiece (30,522 tokens)
- **Sequence length**: **300 tokens** (padded or truncated)
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

### 3.1 Small BERT Encoder

**Model**: `google/bert_uncased_L-4_H-512_A-8` (Small BERT)
- **Transformer layers**: **4 encoder blocks** (vs 12 in BERT-base)
- **Hidden size**: **512** (vs 768 in BERT-base)
- **Attention heads**: **8 per layer** (vs 12 in BERT-base)
- **Parameters**: ~**29M** (vs ~110M in BERT-base)
- **Gradient checkpointing**: **Enabled** (saves GPU memory)

**Key advantages over BERT-base**:
- **3.8× fewer parameters** (29M vs 110M)
- **Faster training** (~50% faster per epoch)
- **Lower memory footprint**
- **Faster inference** for production deployment

### 3.2 Classification Head

**Native HuggingFace head** (`AutoModelForSequenceClassification`):
- **Architecture**: `[CLS] → Dense(18, softmax)`
- **Dropout**: 
  - `hidden_dropout_prob`: **0.3**
  - `attention_probs_dropout_prob`: **0.1**
- **Output**: 18 neurons (one per class)

---

## 4. Training Configuration

### 4.1 Loss Function

**Focal Loss** with class weights and label smoothing:

- **Focal parameter (γ)**: **0.5** (moderate focusing on hard examples)
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
| 1    | 16        | 0.8883  | 0.8873       |
| 2    | 19        | 0.8851  | 0.8833       |
| 3    | 18        | 0.8808  | 0.8796       |

**Average CV F1**: **0.8847**

**Key observations**:
- All folds converged to similar performance (~88% F1)
- Early stopping triggered between epochs 16-19
- Consistent validation accuracy across folds
- Slightly later convergence than BERT-base (16-19 vs 9-11 epochs)

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
| Accuracy  | 87.66% |
| Precision | 0.8834 |
| Recall    | 0.8766 |
| **F1-score** | **0.8780** |

### 6.3 Per-Class Performance

**High-performing classes** (F1 > 0.90):
- Class 3: F1 = 0.9428 (812 samples) - largest class
- Class 16: F1 = 0.9091 (31 samples)
- Class 7: F1 = 0.9355 (30 samples)
- Class 4: F1 = 0.9005 (474 samples)

**Moderate-performing classes** (0.70 < F1 < 0.90):
- Class 1: F1 = 0.8792 (105 samples)
- Class 0: F1 = 0.8333 (12 samples)
- Class 6: F1 = 0.8727 (25 samples)
- Class 8: F1 = 0.8263 (83 samples)
- Class 2: F1 = 0.8205 (20 samples)
- Class 12: F1 = 0.7787 (133 samples)
- Class 5: F1 = 0.7671 (38 samples)
- Class 10: F1 = 0.7589 (99 samples)
- Class 14: F1 = 0.7500 (27 samples)
- Class 9: F1 = 0.7273 (37 samples)
- Class 15: F1 = 0.7317 (19 samples)
- Class 11: F1 = 0.7879 (20 samples)

**Lower-performing classes** (F1 < 0.70):
- Class 13: F1 = 0.6622 (70 samples)
- Class 17: F1 = 0.5882 (7 samples) - smallest class

**Macro average**: Precision 0.7951, Recall 0.8254, F1 0.8040
**Weighted average**: Precision 0.8834, Recall 0.8766, F1 0.8780

---

## 7. Error Analysis

### 7.1 Confusion Matrix

```
Shape: (18, 18)

Predicted →
Actual ↓      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
    0        10   0   0   0   1   0   0   0   0   0   0   0   0   1   0   0   0   0
    1         0  91   1   3   3   0   0   0   1   0   2   0   2   0   0   2   0   0
    2         0   0  16   0   0   0   0   0   0   0   1   0   0   0   0   3   0   0
    3         1   3   0 758  23   2   2   0   4   2   6   0   3   6   1   1   0   0
    4         1   4   0  13 421   0   1   0   1   5  16   0   2   3   3   0   3   1
    5         0   0   0   2   1  28   0   0   0   0   3   0   2   1   1   0   0   0
    6         0   0   0   0   0   0  24   0   0   0   0   0   0   0   0   0   0   1
    7         0   0   0   0   1   0   0  29   0   0   0   0   0   0   0   0   0   0
    8         0   0   1   1   2   0   2   0  69   2   1   0   2   2   0   0   0   1
    9         0   2   0   0   1   0   0   1   0  28   5   0   0   0   0   0   0   0
   10         0   0   0   7   1   0   0   1   1   3  85   0   0   0   1   0   0   0
   11         0   0   0   1   0   0   0   0   0   0   1  13   0   1   3   0   0   1
   12         0   1   0   6   1   2   0   1   5   0   2   0  95  14   3   1   1   1
   13         0   0   0   4   5   3   0   0   1   0   2   0   4  49   1   0   1   0
   14         0   0   0   0   0   0   0   0   1   0   1   0   0   1  24   0   0   0
   15         0   1   1   1   0   0   1   0   0   0   0   0   0   0   0  15   0   0
   16         0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0  30   0
   17         0   0   0   0   1   0   0   0   1   0   0   0   0   0   0   0   0   5
```

### 7.2 Main Error Patterns

**Most confused class pairs**:
1. **Class 3 ↔ Class 4**: 36 misclassifications (23+13)
   - Both are large, dominant classes
   - Likely financial/economic overlap
2. **Class 12 ↔ Class 13**: 18 misclassifications (14+4)
   - Similar topic domains
3. **Class 4 → Class 10**: 16 misclassifications
   - One-directional confusion
4. **Class 10 → Class 3**: 7 misclassifications
   - Related to Class 3's dominance

### 7.3 Error Sources

1. **Remaining class imbalance**:
   - Despite filtering, Class 3 (3,159 samples) dominates
   - Smallest classes (50-70 samples) still show lower F1
   - Class 13 (70 samples): 66.22% F1
   - Class 17 (7 test samples): 58.82% F1

2. **Confusion between similar topics**:
   - Financial and economic news categories overlap
   - Political and business topics confused
   - Class 3 and Class 4 show bidirectional confusion

3. **Sequence truncation**:
   - 300-token limit may cut important context
   - Longer articles lose tail information

4. **Model capacity**:
   - Small BERT has fewer layers (4 vs 12)
   - May struggle with more nuanced distinctions
   - Trade-off between speed and performance

---

## 8. Training Time and Computational Cost

### 8.1 Training Duration

**Cross-Validation** (from training logs):
- Fold 1: ~9 minutes (20 epochs, early stopped at 16)
- Fold 2: ~8.5 minutes (20 epochs, early stopped at 19)
- Fold 3: ~11 minutes (20 epochs, early stopped at 18)

**Total training time**: ~28.5 minutes (~0.48 hours)

**Comparison with BERT-base**:
- BERT-base: ~137 minutes (~2.3 hours)
- Small BERT: ~28.5 minutes (~0.48 hours)
- **Speed-up**: ~4.8× faster training

### 8.2 Training Efficiency

- **Iterations per second**: ~6.3 it/s (vs ~1.2 it/s for BERT-base)
- **Mixed precision**: FP16 enabled
- **Gradient checkpointing**: Enabled (trades compute for memory)
- **Checkpoint management**: Auto-resume capability
- **Memory usage**: Significantly lower than BERT-base

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

## 10. Comparison with Other Implementations

### 10.1 Performance Comparison

| Model | Classes | Accuracy | Precision | Recall | F1-score | Training Time |
|-------|---------|----------|-----------|--------|----------|---------------|
| **Small BERT** | **18** | **87.66%** | **0.8834** | **0.8766** | **0.8780** | **~0.48 hrs** |
| BERT-base | 18 | 87.56% | 0.8817 | 0.8756 | 0.8773 | ~2.3 hrs |

**Performance difference**: +0.10% accuracy, +0.07% F1

**Key findings**:
- Small BERT achieves **comparable performance** to BERT-base
- **4.8× faster training** with minimal accuracy loss
- Excellent trade-off for production deployment
- Validates that smaller models can match larger ones on focused tasks

### 10.2 Architecture Differences

| Aspect | Small BERT | BERT-base |
|--------|-----------|-----------|
| Layers | 4 | 12 |
| Hidden size | 512 | 768 |
| Attention heads | 8 | 12 |
| Parameters | ~29M | ~110M |
| Training speed | ~6.3 it/s | ~1.2 it/s |
| Memory usage | Lower | Higher |

### 10.3 Training Configuration (Same for Both)

| Aspect | Value |
|--------|-------|
| Loss | Weighted focal (γ=0.5) |
| Label smoothing | 0.05 |
| Batch size | 8 (accum 4 = 32) |
| Learning rate | 3e-5 (fixed) |
| Epochs | 20 (early stopping) |
| Patience | 7 |
| Text cleaning | Minimal |

---

## 11. Strengths and Limitations

### 11.1 Strengths

1. **Excellent speed-performance trade-off**:
   - 4.8× faster training than BERT-base
   - Only 0.07% F1 drop compared to BERT-base
   - Ideal for rapid experimentation and iteration

2. **Production-ready**:
   - Lower memory footprint
   - Faster inference
   - Easier deployment on resource-constrained devices
   - Suitable for real-time applications

3. **Minimal preprocessing**:
   - Preserves stopwords, punctuation, numbers
   - Better contextual information retention
   - Less aggressive than BERT-base Reuters implementation

4. **Robust training**:
   - 3-fold CV reduces overfitting
   - Ensemble improves generalization
   - Auto-resume from checkpoints
   - Consistent performance across folds

### 11.2 Limitations

1. **Slightly lower performance ceiling**:
   - 4 layers vs 12 may limit complex pattern learning
   - Macro F1 (0.8040) lower than weighted F1 (0.8780)
   - Struggles more with rare classes

2. **Class imbalance challenges**:
   - Class 3 (3,159 samples) still dominates
   - Smallest classes (50-70 samples) show lower F1
   - Class 17 (7 test samples): 58.82% F1

3. **Reduced coverage**:
   - Only 18 classes (vs original 46)
   - 28 rare classes excluded
   - Not suitable for comprehensive classification

4. **Sequence truncation**:
   - 300-token limit may cut important context
   - Longer articles lose tail information

---

## 12. Future Improvements

### 12.1 Model Enhancements

- **Distillation from BERT-base**: Transfer knowledge from larger model
- **Ensemble with BERT-base**: Combine Small BERT speed with BERT-base accuracy
- **Task-specific pre-training**: Further pre-train on Reuters corpus
- **Adaptive sequence length**: Dynamic padding based on article length

### 12.2 Training Techniques

- **Data augmentation**: Back-translation, paraphrasing for small classes
- **Curriculum learning**: Start with easy examples
- **Mixup/CutMix**: Advanced augmentation
- **Self-training**: Use unlabeled Reuters data

### 12.3 Deployment Optimizations

- **ONNX export**: For production inference
- **Quantization**: INT8 for faster inference
- **TensorRT optimization**: GPU acceleration
- **Model pruning**: Further reduce parameters

### 12.4 Dataset Enhancements

- **Hierarchical classification**: Group similar classes
- **Multi-label approach**: Allow multiple topics per article
- **Active learning**: Focus on hard examples

---

## 13. Conclusion

This implementation demonstrates fine-tuning **Small BERT (L-4_H-512_A-8)** on a **filtered Reuters-21578 dataset** (≥50 samples per class) with advanced techniques:

- **Ensemble F1**: 0.8780 (87.66% accuracy)
- **Training time**: ~0.48 hours (4.8× faster than BERT-base)
- **Model size**: ~29M parameters (3.8× smaller than BERT-base)
- **Performance**: Comparable to BERT-base with minimal loss

**Key achievements**:
- **Excellent speed-performance trade-off**: 4.8× faster with only 0.07% F1 drop
- **Production-ready**: Lower memory, faster inference
- **Robust training**: Consistent 88% F1 across all folds
- **Minimal preprocessing**: Better information retention

**Trade-offs**:
- **Slightly lower macro F1**: 0.8040 vs 0.8064 (BERT-base)
- **Reduced model capacity**: 4 layers vs 12 layers
- **Reduced coverage**: 18 classes vs 46 original classes

**Best use case**: Production systems requiring **fast training and inference** for major news categories where speed is critical and a small performance trade-off is acceptable.

**Key takeaway**: Small BERT achieves **87.66% accuracy** on Reuters classification with **4.8× faster training** than BERT-base, demonstrating that smaller transformer models can deliver competitive performance on focused classification tasks while being significantly more efficient for production deployment.
