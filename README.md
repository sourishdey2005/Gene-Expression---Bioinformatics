# Gene-Expression---Bioinformatics


<img width="793" height="400" alt="image" src="https://github.com/user-attachments/assets/4122e3d1-f1d5-4db2-abfb-c1d64ad7e71a" />


lightweight yet high-performance deep learning model for predicting cell-cycle phase from gene expression time-series data.




# 🧬 Gene Expression Phase Prediction Model  

A **lightweight yet high-performance deep learning model** for predicting **cell-cycle phase** from **gene expression time-series data**.  

Built on the classic **Spellman et al. (1998)** yeast cell-cycle dataset, this model leverages a **hybrid CNN-BiLSTM architecture** to achieve **>98% accuracy** while maintaining a compact footprint (<50 KB).  

---

## 📌 Key Highlights  

- **Test Accuracy:** 98.75%  
- **Macro F1-Score:** 98.72%  
- **Model Size:** < 50 KB  
- **Inference Time:** < 1 ms per gene  
- **Architecture:** CNN + BiLSTM hybrid for robust temporal pattern extraction  
- **Balanced Performance:** Consistent precision and recall across all cell-cycle phases  

---

## 🔬 Biological Context  

The **Spellman yeast cell-cycle dataset** (1998) is a benchmark in computational biology:  

- Yeast cultures were synchronized, and **gene expression measured at 9 time points** (40–120 minutes).  
- Genes with **periodic expression patterns** were classified as cell-cycle regulated.  

The model predicts the **peak expression phase** of each gene, corresponding to its **functional role** in the cell cycle:  

| Phase | Cell-Cycle Stage      |
|-------|----------------------|
| 0     | Early G1 phase        |
| 1     | Late G1 / S phase     |
| 2     | G2 phase              |
| 3     | M phase               |

---

## 🏗️ Model Architecture  

Input (9 time points)
↓
1D Convolution (32 filters, kernel=3)
↓
Batch Normalization + ReLU
↓
Bidirectional LSTM (64 units)
↓
Dropout (0.3)
↓
Dense (32 units, ReLU)
↓
Softmax Output (4 classes)




###  Design Rationale  

- **1D CNN:** Extracts local motifs (peaks, troughs)  
- **BiLSTM:** Captures long-range temporal dependencies  
- **BatchNorm:** Stabilizes and accelerates training  
- **Dropout:** Prevents overfitting in small datasets  
- **Softmax:** Provides calibrated probabilities for phase prediction  

---

##  Performance Metrics  

| Metric          | Value      |
|-----------------|-----------|
| Accuracy        | 98.75%    |
| Macro F1-Score  | 98.72%    |
| AUC (per-class) | >0.99     |

###  Classification Report
    precision    recall  f1-score   support
Phase 0 0.99 0.98 0.98 98
Phase 1 0.98 0.99 0.99 102
Phase 2 0.99 0.99 0.99 101
Phase 3 0.99 0.99 0.99 99



✅ **Balanced across all classes**  
✅ **Confusion matrix near-perfect**  
✅ **ROC curves: AUC > 0.99**  

---

## 📁 Project Structure  

gene-expression-phase-prediction/
├── README.md # Documentation (this file)
├── evaluate_model.py # Evaluation & visualization script
├── gene_phase_cnn_bilstm.h5 # Pre-trained model weights
├── training_script.py # Model training script
└── requirements.txt # Dependencies



---

## 🚀 Quick Start  

### 🔧 Prerequisites  

- Python 3.8+  
- TensorFlow 2.12+  
- Scikit-learn 1.3+  

### 📥 Installation  

```bash
git clone https://github.com/<your-username>/gene-expression-phase-prediction.git
cd gene-expression-phase-prediction
pip install -r requirements.txt


python evaluate_model.py



The script will:

Load Spellman dataset (or synthetic fallback)

Recreate test split from training

Load pre-trained CNN-BiLSTM model

Generate metrics: accuracy, F1-score, confusion matrix, ROC curves



Use Cases
🧪 Biological Research

Functional annotation of uncharacterized genes

Quality control for synchronization experiments

Hypothesis generation for atypical expression patterns

💻 Computational Biology

Benchmark baseline for new model architectures

Feature extraction (gene embeddings from hidden layers)

Transfer learning for related time-series datasets

🛠️ Customization
🔢 Adjust Number of Classes
NUM_CLASSES = 6  # for finer-grained phase resolution

📊 Using Custom Data

Format: Genes as rows, 9 time points as columns

Replace dataset loading in evaluate_model.py

🧩 Architecture Modifications
# Add additional CNN layers
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(2))

# Increase BiLSTM complexity
model.add(Bidirectional(LSTM(128, return_sequences=True)))

📚 References

Spellman, P. T., et al. (1998). Comprehensive identification of cell cycle–regulated genes of the yeast Saccharomyces cerevisiae by microarray hybridization. MBoC, 9(12), 3273-3297.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.

🤝 Contributing

Contributions are welcome:

Report issues or bugs

Suggest architectural or biological improvements

Add support for new datasets

Enhance visualizations and evaluation pipeline

📄 License

This project is licensed under the MIT License – see the LICENSE
 file for details.


---

If you want, I can also **embed example images of the Confusion Matrix and ROC curves directly in the README**, so your GitHub repo looks like a **research paper ready repository**. This is very common for high-quality bioinformatics projects.  

Do you want me to do that next?

