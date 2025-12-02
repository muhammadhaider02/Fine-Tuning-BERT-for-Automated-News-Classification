# Fine-Tuning BERT for Automated News Classification - Original Paper Summary

This document summarizes the implementation described in the paper *“Fine-Tuning BERT for Automated News Classification”* by Salih et al. The goal is to capture the dataset setup, model architecture, training configuration, and results as stated in the paper.

---

## 1. Problem Definition

- Task: Automatic **news topic classification** for English news articles.
- Domain: General news articles from the **Reuters-21578** collection.
- Objective: Show that a **fine-tuned BERT model** can outperform:
  - Traditional machine learning models such as Naive Bayes (NB), Support Vector Machine (SVM), and Random Forest (RF).
  - A **non-fine-tuned BERT** baseline.
  - A **CNN-LSTM hybrid** deep learning model.

The main claim is that transfer learning with a fine-tuned BERT model achieves significantly better performance on Reuters news classification than these baselines.

---

## 2. Dataset and Preprocessing

### 2.1 Reuters-21578 Dataset

- Corpus: **11,228** news articles.
- Categories: **46** topic classes.
- Labels: Each document can belong to one or more topic categories, but the problem is treated as a **multi-class** classification task in this work.
- Split: The dataset is divided into **80% training** and **20% testing**.

The Reuters-21578 dataset is widely used as a benchmark corpus in information retrieval and text classification.

### 2.2 Vocabulary Truncation and Sequence Length

- Vocabulary is **truncated to the top 10,000 most frequent words**.
- After tokenization, each article is **padded or truncated to 300 tokens**.
- The sequence length **300** is chosen based on the article length distribution so that most news items fit within this length without heavy truncation.

### 2.3 From Index Sequences to BERT Inputs

The dataset originally provides **sequences of word indices** rather than raw text. The paper describes the following pipeline:

1. **Text decoding**  
   - Use a **word-index mapping** to reconstruct the text form of each article from its integer sequence.
   - The decoded text is then tokenized using **BERT’s pre-trained WordPiece tokenizer**.
   - WordPiece allows the model to handle out-of-vocabulary terms by splitting them into subword units.

2. **Padding and truncation**  
   - All tokenized sequences are **padded or cut to 300 tokens**.
   - This ensures uniform length for all samples, which is required by BERT.

3. **BERT input representation**  
   For each article, the following inputs are constructed:
   - **Token IDs**: Integer IDs from BERT’s vocabulary for each subword token.
   - **Attention masks**: Binary mask where 1 indicates a real token and 0 indicates padding.
   - **Token type IDs (segment IDs)**: Since this is a **single-sentence classification** task, all token type IDs are set to 0.
   - A **[CLS] token** is added at the beginning and a **[SEP] token** is added at the end of each sequence.

The model uses these inputs to feed the articles into BERT.

---

## 3. Model Architecture (As Described in the Paper)

### 3.1 BERT Encoder

The authors use **Small BERT (L-4_H-512_A-8)** rather than BERT-base. This variant has:

- **4 transformer layers** (encoder blocks).
- **Hidden size = 512**.
- **8 self-attention heads** per layer.

Small BERT is chosen to reduce computational cost while still capturing contextual information effectively. It supports faster fine-tuning and inference and is more suitable for resource-constrained environments.

Each transformer layer contains:

- **Multi-Head Self-Attention (MHSA)**  
  - Every token can attend to all other tokens in the sequence.
  - Multiple attention heads focus on different aspects of context.
- **Feed-Forward Network (FFN)**  
  - A two-layer fully connected network with nonlinearity.
  - Combined with residual connections and layer normalization to stabilize training.

The **pooled output of the [CLS] token** after the final transformer layer is used as the document-level representation for classification.

### 3.2 Task-Specific Classification Head

On top of the BERT encoder, the authors add fully connected layers tailored to the Reuters classification task:

1. **Dense layer**  
   - Size: **256 neurons**.  
   - Activation: **ReLU**.  
   - Purpose: Learn higher level semantic features from the [CLS] representation.

2. **Dropout layer**  
   - Dropout rate: **0.3**.  
   - Motivation: Regularization to reduce overfitting by randomly deactivating 30% of the units during training.

3. **Output layer**  
   - Size: **46 neurons**, one per Reuters class.  
   - Activation: **softmax**.  
   - Output: A probability distribution over all 46 categories that sums to 1.

So the classification head can be summarized as:

`[CLS] → Dense(256, ReLU) → Dropout(0.3) → Dense(46, softmax)`

---

## 4. Training Configuration

### 4.1 Loss Function and Optimizer

- **Loss function**: **Sparse categorical cross-entropy**  
  - Appropriate because labels are integer-encoded class indices.
- **Optimizer**: **AdamW**  
  - Adam with weight decay.  
  - Weight decay regularizes the model by penalizing large weights and helps generalization.

The paper emphasizes the use of weight decay through AdamW but does not specify the exact learning rate, warmup schedule, or decay values.

### 4.2 Hardware and Training Setup

- Hardware: **NVIDIA RTX 4060 GPU**.
- Batch size: **32**.
- Number of epochs: **10**.
- Reasoning: They observed that performance improvements **plateau after about 10 epochs**, and training longer increases the risk of overfitting without significant gains.

The sequence length 300 and the use of Small BERT further help keep GPU memory usage manageable.

---

## 5. Evaluation and Results

### 5.1 Metrics

The paper evaluates the model using:

- **Accuracy**.
- **Precision**.
- **Recall**.
- **F1-score**.

These are reported for the proposed fine-tuned BERT model and for all baseline models.

### 5.2 Model Comparison (Table I)

The paper compares six models:

- Fine-tuned BERT (proposed).
- Non-fine-tuned BERT.
- Naive Bayes (NB).
- Support Vector Machine (SVM).
- Random Forest (RF).
- CNN-LSTM hybrid.

The reported performance on Reuters-21578 is:

- **Fine-tuned BERT (proposed model)**  
  - Accuracy: **91.77%**  
  - Precision: **0.92**  
  - Recall: **0.91**  
  - F1-score: **0.915**

- **Non-fine-tuned BERT**  
  - Accuracy: **83.45%**  
  - Precision: **0.84**  
  - Recall: **0.83**  
  - F1-score: **0.835**

- **Naive Bayes (NB)**  
  - Accuracy: **76.32%**  
  - Precision: **0.78**  
  - Recall: **0.75**  
  - F1-score: **0.765**

- **Support Vector Machine (SVM)**  
  - Accuracy: **80.19%**  
  - Precision: **0.81**  
  - Recall: **0.79**  
  - F1-score: **0.80**

- **Random Forest (RF)**  
  - Accuracy: **78.41%**  
  - Precision: **0.79**  
  - Recall: **0.77**  
  - F1-score: **0.78**

- **CNN-LSTM hybrid**  
  - Accuracy: **85.67%**  
  - Precision: **0.86**  
  - Recall: **0.85**  
  - F1-score: **0.856**

The fine-tuned BERT model clearly **outperforms all baselines**, including the CNN-LSTM hybrid, in both accuracy and F1-score.

---

## 6. Error Analysis

The paper identifies three main sources of errors for the fine-tuned BERT model:

1. **Ambiguous multi-topic articles**  
   - Some news items cover more than one topic (for example, finance and politics).  
   - The model is forced to choose a single label, so multi-label classification could improve this in the future.

2. **Underrepresented categories**  
   - Several Reuters classes have very few examples.  
   - The model has difficulty learning robust patterns for these rare topics, which leads to higher error rates in those classes.

3. **Long articles and truncation**  
   - The 300-token limit means long articles are truncated.  
   - Important information at the end of an article can be cut off, reducing classification quality.

They suggest future work on multi-label classification, data augmentation or class weighting for rare classes, and possibly document-level or hierarchical transformer models for long texts.

---

## 7. Computational Cost and Trade-Offs

The paper explicitly discusses the **computational cost** of fine-tuning BERT compared to other models.

### 7.1 Training Time Comparison (Table II)

Approximate training times on GPU:

- Fine-tuned BERT: **~3 hours**.
- Non-fine-tuned BERT: **~1 hour**.
- CNN-LSTM hybrid: **~2 hours**.
- NB: **~10 minutes**.
- SVM: **~45 minutes**.
- RF: **~30 minutes**.

This shows that while fine-tuned BERT offers the best performance, it is also the **most expensive model to train**.

### 7.2 Deployment Considerations

- BERT models require **more memory and compute** than NB, SVM, or RF.
- The authors mention techniques such as:
  - **Model pruning**.
  - **Quantization**.
  - **Knowledge distillation**.
  - **Low-Rank Adaptation (LoRA)** style methods.

These approaches could help make transformer models more efficient for real-time or low-resource deployment.

---

## 8. Conclusion of the Original Paper

- Fine-tuned BERT on Reuters-21578 reaches **91.77% accuracy**, outperforming traditional classifiers, non-fine-tuned BERT, and a CNN-LSTM hybrid model.
- The gains come from:
  - BERT’s ability to capture **bidirectional context**.
  - Effective transfer learning from large-scale pre-training to a smaller news dataset.
- Main trade-off:
  - **Higher accuracy and F1** at the cost of **greater computational complexity and training time**.
- Future directions mentioned:
  - Lightweight transformer variants for real-time systems (such as DistilBERT).
  - Multi-label classification for multi-topic articles.
  - Combined architectures that integrate BERT with CNN or LSTM layers.
  - Improved methods to handle long articles and class imbalance.

This completes the faithful summary of the original paper’s implementation and results.
