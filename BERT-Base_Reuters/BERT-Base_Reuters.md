# Fine-Tuning BERT-Base for Reuters News Classification - Implementation Report

This document describes the implementation of BERT-base fine-tuning on the Reuters-21578 dataset with advanced training techniques including focal loss, class weighting, label smoothing, and 3-fold cross-validation ensemble.

---

## 1. Problem Definition

- **Task**: Automatic **news topic classification** for English news articles.
- **Domain**: General news articles from the **Reuters-21578** collection.
- **Objective**: Fine-tune **BERT-base-uncased** with advanced training techniques to achieve robust classification performance through:
  - Class-weighted focal loss to handle class imbalance.
  - Label smoothing for better generalization.
  - 3-fold stratified cross-validation with ensemble prediction.
  - Learning rate search for optimal hyperparameters.

---

## 2. Dataset and Preprocessing

### 2.1 Reuters-21578 Dataset

- **Corpus**: **11,228** news articles total.
  - Training samples: **8,982**
  - Test samples: **2,246**
- **Categories**: **46** topic classes.
- **Split**: 80% training / 20% testing (using Keras default split with seed=113).

### 2.2 Vocabulary and Sequence Length

- **Vocabulary**: Truncated to the top **10,000 most frequent words**.
- **Sequence length**: **300 tokens** (padded or truncated).
- **Special tokens**:
  - `[PAD]` (index 0)
  - `[START]` (index 1)
  - `[UNK]` (index 2) for out-of-vocabulary words
  - `[UNUSED]` (index 3)

### 2.3 Text Decoding and Tokenization

1. **Text decoding**:
   - Original Reuters data provides sequences of word indices.
   - Word-index mapping downloaded from TensorFlow datasets.
   - Sequences decoded to raw text using reverse word index.

2. **BERT tokenization**:
   - **Tokenizer**: `bert-base-uncased` WordPiece tokenizer.
   - **Max length**: 300 tokens.
   - **Padding**: `max_length` strategy.
   - **Truncation**: Enabled.

3. **BERT inputs**:
   - `input_ids`: Token IDs from BERT vocabulary.
   - `attention_mask`: Binary mask (1 for real tokens, 0 for padding).
   - `token_type_ids`: All zeros (single-sentence classification).

### 2.4 Class Imbalance Handling

**Class weights** computed using inverse-frequency weighting:
- Formula: `weight[c] = 1 / max(count[c], 1)`
- Weights clipped to range [0, 10] for stability.
- Normalized so sum equals number of classes.

Example weights (tensor):
```
[0.5730, 0.0729, 0.4258, 0.0100, 0.0162, 1.8537, 0.6565, 1.9695, ...]
```

---

## 3. Model Architecture

### 3.1 BERT Encoder

**Model**: `bert-base-uncased` (12-layer, 768-hidden, 12-heads)
- **Transformer layers**: 12 encoder blocks.
- **Hidden size**: 768.
- **Attention heads**: 12 per layer.
- **Parameters**: ~110M.

Each transformer layer contains:
- **Multi-Head Self-Attention (MHSA)**: Bidirectional context modeling.
- **Feed-Forward Network (FFN)**: Two-layer MLP with GELU activation.
- **Residual connections** and **layer normalization**.

The **pooled output of the [CLS] token** is used as the document representation.

### 3.2 Custom Classification Head

Architecture: `[CLS] → Dropout → Dense(256) → ReLU → Dropout → Dense(46)`

1. **First dropout layer**:
   - Dropout rate: **0.3**.
   - Applied to [CLS] pooled output (768-dim).

2. **Intermediate dense layer**:
   - Size: **256 neurons**.
   - Activation: **ReLU**.
   - Purpose: Learn task-specific features.

3. **Second dropout layer**:
   - Dropout rate: **0.3**.
   - Regularization before final classification.

4. **Output layer**:
   - Size: **46 neurons** (one per class).
   - Activation: **Softmax** (applied in loss function).

---

## 4. Training Configuration

### 4.1 Loss Function

**Weighted Focal Loss** with label smoothing:

- **Base**: Multi-class focal loss.
- **Focal parameter (γ)**: 1.0 (reduced from 2.0 for stability).
- **Class weights**: Applied per-class (α weights).
- **Label smoothing**: 0.1 (10% smoothing).

Formula:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t` is the probability of the true class.
- `α_t` is the class weight.
- `γ` controls focus on hard examples.

Label smoothing creates soft targets:
```
y_smooth = (1 - ε) * y_true + ε / num_classes
```

### 4.2 Optimizer and Scheduler

- **Optimizer**: **AdamW** (Adam with weight decay).
- **Weight decay**: 0.01.
- **Learning rate**: **3e-5** (selected via LR search).
- **LR scheduler**: **Cosine** with warmup.
- **Warmup ratio**: 0.15 (15% of total steps).
- **Gradient clipping**: Max norm 1.0.

### 4.3 Training Hyperparameters

- **Batch size**: 16 per device.
- **Gradient accumulation steps**: 2 (effective batch size = 32).
- **Epochs**: 20 (with early stopping).
- **Early stopping patience**: 7 epochs.
- **Early stopping metric**: Validation F1-score.
- **Min delta**: 1e-4.
- **Mixed precision**: FP16 enabled.

### 4.4 Hardware

- **Device**: CUDA GPU (forced, no CPU fallback).
- **GPU**: NVIDIA GPU with CUDA support.

---

## 5. Training Strategy

### 5.1 Phase 6: Learning Rate Search

**Objective**: Find optimal learning rate.

**Setup**:
- Single stratified split: 80% train / 20% validation.
- Candidates tested: **2e-5**, **3e-5**.
- Epochs: 20 per candidate.

**Results**:
| Learning Rate | Best Val F1 | Best Epoch |
|--------------|-------------|------------|
| 2e-5         | 0.8617      | 16         |
| **3e-5**     | **0.8656**  | **10**     |

**Selected**: **3e-5** (best validation F1).

### 5.2 Phase 7: 3-Fold Stratified Cross-Validation

**Objective**: Train robust ensemble of models.

**Setup**:
- **Folds**: 3 stratified folds.
- **Learning rate**: 3e-5 (from LR search).
- **Epochs**: 20 per fold (with early stopping).
- **Validation**: Each fold uses different 1/3 of data.

**Fold Results**:
| Fold | Best Epoch | Val F1  | Val Accuracy |
|------|-----------|---------|--------------|
| 1    | 18        | 0.8548  | 0.8400       |
| 2    | 18        | 0.8584  | 0.8504       |
| 3    | 13        | 0.8550  | 0.8457       |

**Average CV F1**: 0.8561

---

## 6. Evaluation and Results

### 6.1 Metrics

- **Accuracy**
- **Precision** (weighted average)
- **Recall** (weighted average)
- **F1-score** (weighted average)

### 6.2 Ensemble Test Performance

**Method**: Average logits from 3 fold models, then argmax.

**Test Set Results** (held-out 2,246 samples):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 83.70% |
| Precision | 0.8708 |
| Recall    | 0.8370 |
| **F1-score** | **0.8497** |

### 6.3 Per-Class Performance

**High-performing classes** (F1 > 0.90):
- Class 3: F1 = 0.9222 (813 samples) - largest class
- Class 4: F1 = 0.9003 (474 samples)
- Class 10: F1 = 0.9333 (30 samples)
- Class 6: F1 = 0.9032 (14 samples)
- Class 32: F1 = 0.9474 (10 samples)

**Low-performing classes** (F1 < 0.50):
- Class 5: F1 = 0.0000 (5 samples) - extremely rare
- Class 35: F1 = 0.0952 (6 samples)
- Class 40: F1 = 0.3529 (10 samples)
- Class 27: F1 = 0.3636 (4 samples)
- Class 33: F1 = 0.4000 (5 samples)

**Macro average**: Precision 0.6716, Recall 0.8027, F1 0.7074
**Weighted average**: Precision 0.8708, Recall 0.8370, F1 0.8497

---

## 7. Error Analysis

### 7.1 Main Error Sources

1. **Extreme class imbalance**:
   - Classes with < 10 samples show very poor performance.
   - Class 5 (5 samples): 0% precision/recall.
   - Class 35 (6 samples): 5.26% precision.
   - Even with class weights, insufficient data for rare classes.

2. **Confusion between similar topics**:
   - Financial news categories often confused.
   - Political and economic topics overlap.

3. **Sequence truncation**:
   - 300-token limit may cut important context.
   - Longer articles lose tail information.

### 7.2 Confusion Matrix Insights

- **Shape**: 46 × 46
- **Diagonal dominance**: Most predictions correct for major classes.
- **Off-diagonal errors**: Concentrated in rare classes and semantically similar categories.

---

## 8. Training Time and Computational Cost

### 8.1 Training Duration

**Learning Rate Search**:
- 2e-5: 124.6 minutes (20 epochs)
- 3e-5: 122.1 minutes (17 epochs, early stopped)

**Cross-Validation**:
- Fold 1: 90.0 minutes (20 epochs)
- Fold 2: 172.3 minutes (20 epochs)
- Fold 3: 108.3 minutes (20 epochs, early stopped)

**Total training time**: ~617 minutes (~10.3 hours)

### 8.2 Training Efficiency

- **Samples per second**: ~11-22 (varies by fold)
- **Steps per second**: ~0.36-0.70
- **Mixed precision**: FP16 enabled for faster training
- **Gradient checkpointing**: Not used (sufficient GPU memory)

---

## 9. Key Implementation Details

### 9.1 Custom Trainer

**WeightedFocalTrainer** extends HuggingFace `Trainer`:
- Implements focal loss with class weights.
- Applies label smoothing.
- Caches class weights on device.

### 9.2 Callbacks

**EpochMetricsAndEarlyStoppingCallback**:
- Evaluates on validation set after each epoch.
- Prints epoch metrics (loss, accuracy, F1, time).
- Saves best model based on validation F1.
- Implements early stopping with patience.
- Tracks total training time.

### 9.3 Model Saving and Loading

- **Format**: SafeTensors (`.safetensors`)
- **Best model**: Saved per fold in `best/` subdirectory
- **Loading**: Custom state dict loading for ensemble

---

## 10. Comparison with Original Paper

### 10.1 Architecture Differences

| Aspect | Original Paper | This Implementation |
|--------|---------------|---------------------|
| BERT variant | Small BERT (L-4, H-512, A-8) | BERT-base (L-12, H-768, A-12) |
| Intermediate layer | 256 neurons | 256 neurons |
| Dropout | 0.3 | 0.3 (applied twice) |
| Activation | ReLU | ReLU |

### 10.2 Training Differences

| Aspect | Original Paper | This Implementation |
|--------|---------------|---------------------|
| Loss | Sparse categorical cross-entropy | Weighted focal loss + label smoothing |
| Optimizer | AdamW | AdamW |
| Epochs | 10 | 20 (with early stopping) |
| Batch size | 32 | 32 (effective, via accumulation) |
| Validation | Single split | 3-fold cross-validation |
| Ensemble | No | Yes (3 models) |

### 10.3 Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Original Paper (Small BERT) | 91.77% | 0.92 | 0.91 | 0.915 |
| This Implementation (BERT-base ensemble) | 83.70% | 0.8708 | 0.8370 | 0.8497 |

**Note**: Lower performance likely due to:
- Different BERT variant (base vs. small).
- More aggressive regularization (focal loss, label smoothing).
- Different random seeds and data splits.
- Ensemble averaging may smooth predictions.

---

## 11. Strengths and Limitations

### 11.1 Strengths

1. **Robust training**:
   - 3-fold CV reduces overfitting to single split.
   - Ensemble improves generalization.

2. **Class imbalance handling**:
   - Focal loss focuses on hard examples.
   - Class weights balance rare classes.

3. **Regularization**:
   - Label smoothing prevents overconfidence.
   - Dropout (0.3) in two layers.
   - Weight decay (0.01).

4. **Reproducibility**:
   - Fixed seeds (42, 113).
   - Deterministic data loading.

### 11.2 Limitations

1. **Rare class performance**:
   - Classes with < 10 samples still perform poorly.
   - Focal loss and weights insufficient for extreme imbalance.

2. **Computational cost**:
   - 10+ hours training time.
   - 3× model storage (ensemble).

3. **Sequence length**:
   - 300 tokens may truncate long articles.
   - Important information could be lost.

4. **Single-label assumption**:
   - Reuters articles can have multiple topics.
   - Forced single-label classification.

---

## 12. Future Improvements

### 12.1 Model Architecture

- **Hierarchical attention**: Handle longer documents.
- **Multi-label classification**: Allow multiple topics per article.
- **Lightweight variants**: DistilBERT, TinyBERT for faster inference.

### 12.2 Training Techniques

- **Data augmentation**: Back-translation, synonym replacement for rare classes.
- **Oversampling**: SMOTE or class-balanced sampling.
- **Curriculum learning**: Start with easy examples, progress to hard.

### 12.3 Hyperparameter Tuning

- **Focal gamma**: Test γ ∈ {0.5, 1.0, 2.0, 3.0}.
- **Label smoothing**: Test ε ∈ {0.05, 0.1, 0.15}.
- **Dropout**: Test rates ∈ {0.1, 0.2, 0.3, 0.4}.

### 12.4 Deployment

- **Model compression**: Quantization (INT8), pruning.
- **Knowledge distillation**: Distill ensemble into single model.
- **ONNX export**: For production inference.

---

## 13. Conclusion

This implementation demonstrates fine-tuning **BERT-base** on Reuters-21578 with advanced techniques:

- **Ensemble F1**: 0.8497 (83.70% accuracy)
- **Robust training**: 3-fold CV with early stopping
- **Class imbalance**: Focal loss + class weights + label smoothing

While performance is lower than the original paper's Small BERT (91.77%), this implementation provides:
- More robust evaluation through cross-validation
- Better handling of class imbalance
- Production-ready ensemble approach

The main challenge remains **extreme class imbalance** (classes with < 10 samples), which requires data augmentation or multi-task learning approaches.

**Key takeaway**: BERT-base with proper regularization and ensemble methods achieves strong performance on news classification, but rare classes remain a significant challenge.
