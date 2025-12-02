# Efficient News Topic Classification: A BERT Fine-Tuning Framework for Balanced and Imbalanced Corpora

A comprehensive comparison of BERT variants for news topic classification, featuring implementations on AG News and Reuters-21578 datasets with advanced training techniques including focal loss, label smoothing, and 3-fold cross-validation ensemble.

---

## Performance Comparison

| Model | Dataset | Classes | Accuracy | F1-Score | Training Time | Details |
|-------|---------|---------|----------|----------|---------------|---------|
| **BERT-Base** | AG News | 4 | **94.22%** | **0.9423** | ~2-3 hrs | [📄 Report](BERT-Base_AG-News/BERT-Base_AG-News.md) |
| BERT-Base | Reuters (Full) | 46 | 83.70% | 0.8497 | ~10.3 hrs | [📄 Report](BERT-Base_Reuters/BERT-Base_Reuters.md) |
| BERT-Base | Reuters (Filtered) | 18 | 87.56% | 0.8773 | ~2.3 hrs | [📄 Report](BERT-Base_Reuters-Filtered/BERT-Base_Reuters-Filtered.md) |
| **Small BERT** | Reuters (Filtered) | 18 | 87.66% | 0.8780 | **~0.48 hrs** | [📄 Report](Small-BERT_Reuters_Fine-tuned/Small-BERT_Reuters_Fine-tuned.md) |

---

## Key Findings

### Best Overall Performance
- **BERT-Base on AG News**: 94.22% accuracy, 0.9423 F1
- Balanced 4-class dataset with clear category boundaries
- Excellent for general news categorization (World, Sports, Business, Sci/Tech)

### Best Speed-Performance Trade-off
- **Small BERT on Reuters (Filtered)**: 87.66% accuracy in 0.48 hours
- **4.8× faster** than BERT-Base with only 0.10% accuracy difference
- Ideal for production deployment with resource constraints

### Best for Imbalanced Data
- **BERT-Base Reuters (Filtered)**: 87.56% accuracy on 18 classes
- Aggressive text preprocessing (stopword removal, punctuation cleaning)
- +3.86% accuracy improvement over full 46-class Reuters

### Model Comparison Insights
- **AG News vs Reuters**: AG News achieves higher accuracy (94.22% vs 87.56%) due to balanced dataset and fewer classes
- **Full vs Filtered Reuters**: Filtering ultra-rare classes (<50 samples) improves accuracy by 3.86%
- **BERT-Base vs Small BERT**: Small BERT matches BERT-Base performance with 3.8× fewer parameters (29M vs 110M)
- **Training Time**: Ranges from 0.48 hours (Small BERT) to 10.3 hours (BERT-Base Full Reuters)

---

## Repository Structure

```
Fine-Tuning-BERT-for-Automated-News-Classification/
│
├── BERT-Base_AG-News/
│   ├── BERT-Base_AG-News.md          # Implementation report for AG News
│   ├── BERT-Base_AG-News.py          # Training script
│   └── figures/                      # Confusion matrix, loss plots, accuracy charts
│
├── BERT-Base_Reuters/
│   ├── BERT-Base_Reuters.md          # Implementation report for Reuters (46 classes)
│   └── BERT-Base_Reuters.py          # Training script with LR search + 3-fold CV
│
├── BERT-Base_Reuters-Filtered/
│   ├── BERT-Base_Reuters-Filtered.md # Implementation report for filtered Reuters (18 classes)
│   └── BERT-Base_Reuters-Filtered.py # Training script with aggressive preprocessing
│
├── Small-BERT_Reuters_Fine-tuned/
│   ├── Small-BERT_Reuters_Fine-tuned.md  # Implementation report for Small BERT
│   └── Small-BERT_Reuters_Fine-tuned.py  # Lightweight model training script
│
├── Small-BERT_Reuters_Base-Paper/
│   └── Small-BERT_Reuters_Base-Paper.md  # Original paper summary
│
└── README.md
```

### Implementation Details

Each implementation folder contains:
- **`.md` Report**: Comprehensive documentation including dataset preprocessing, model architecture, training configuration, results, error analysis, and future improvements
- **`.py` Script**: Complete training pipeline with data loading, tokenization, custom loss functions, callbacks, cross-validation, and ensemble evaluation
- **`figures/`** (AG News only): Visualizations including confusion matrices, training/validation loss curves, and per-class accuracy charts

### Key Techniques Used

- **Focal Loss**: Handles class imbalance by focusing on hard-to-classify examples
- **Label Smoothing**: Prevents overconfidence and improves generalization
- **Class Weighting**: Inverse-frequency weighting for imbalanced datasets
- **3-Fold Stratified Cross-Validation**: Robust model evaluation with ensemble prediction
- **Gradient Checkpointing**: Memory-efficient training for larger models
- **Early Stopping**: Prevents overfitting with patience-based monitoring

---

## Quick Reference

### When to Use Each Model

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| **General news categorization** | BERT-Base AG News | Highest accuracy (94.22%), balanced dataset |
| **Production deployment** | Small BERT Reuters | 4.8× faster, minimal accuracy loss |
| **Imbalanced multi-class** | BERT-Base Reuters Filtered | Best handling of class imbalance |
| **Research/experimentation** | BERT-Base Reuters Full | Comprehensive 46-class classification |

### Dataset Characteristics

| Dataset | Classes | Train Samples | Test Samples | Balance | Avg Length |
|---------|---------|---------------|--------------|---------|------------|
| AG News | 4 | 120,000 | 7,600 | Perfect | Short |
| Reuters (Full) | 46 | 8,982 | 2,246 | Imbalanced | Medium |
| Reuters (Filtered) | 18 | 8,227 | 2,042 | Imbalanced | Medium |

---

## Citation

If you use this work, please cite the original paper:

```bibtex
@article{salih2024finetuning,
  title={Fine-Tuning BERT for Automated News Classification},
  author={Salih et al.},
  year={2024}
}
```

---
