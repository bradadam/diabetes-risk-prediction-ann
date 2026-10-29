# 🩺 Clinical Diabetes Risk Prediction — EDA & Deep Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Viz-4C9BE8)

An end-to-end **Healthcare Machine Learning & Deep Learning pipeline** designed to predict diabetes status from **100,000 electronic health records (EHR)**. The repository is structured into two modular stages: **1) Data Preprocessing & Exploratory Data Analysis (EDA)** and **2) Cost-Sensitive Artificial Neural Network (ANN) Modeling** optimized for high clinical sensitivity (recall).

---

## 📌 Project Highlights

* **Clinical Data Hygiene & Preprocessing:**
  * Deduplicated **3,854** redundant EHR entries and filtered anomalous demographic categories.
  * Consolidated fragmented smoking history records (`6` raw labels $\rightarrow$ `4` standardized clinical groups: `never`, `former`, `current`, `No Info`).
  * **IQR Winsorization:** Capped extreme `bmi` outliers at $[Q_1 - 1.5\text{ IQR},\ Q_3 + 1.5\text{ IQR}]$ to preserve sample size without distorting gradient updates.
* **Severe Class Imbalance Handling ($91.5\%$ vs. $8.5\%$):**
  * Stratified $70\% / 15\% / 15\%$ Train/Validation/Test splitting.
  * Applied inverse-frequency **Balanced Class Weights** (`~0.55` for Non-Diabetic vs. `~5.50` for Diabetic) during binary cross-entropy optimization.
* **Regularized Deep Architecture (Keras Functional API):**
  * Multi-layer Perceptron with **Batch Normalization**, **ReLU** activations, and **Dropout ($p = 0.30$)**, monitored via `EarlyStopping` (on `val_auc`) and `ReduceLROnPlateau`.
* **Clinically Tuned Decision Threshold:**
  * Evaluates classification performance using a sensitivity-prioritized probability cutoff to minimize False Negatives (undiagnosed diabetic patients).

---

## 📊 Stage 1: Exploratory Data Analysis (EDA)

### 1. Target Imbalance ($91.5\%$ Non-Diabetic vs. $8.5\%$ Diabetic)
![Class Distribution](assets/class_distribution.png)

### 2. Biomarker Distributions (`age`, `bmi`, `HbA1c_level`, `blood_glucose_level`)
Diabetic patients exhibit distinct right-shifted distributions in **$\text{HbA1c}$** and **Blood Glucose**, whereas **BMI** and **Age** show broader overlap:
![Numerical Distributions](assets/numerical_distributions.png)

### 3. Feature Correlation Matrix
**Blood Glucose Level** and **$\text{HbA1c}$ Level** show the strongest positive Pearson correlation with diabetes diagnosis, followed by **Age**, **BMI**, and **Hypertension**:
![Correlation Heatmap](assets/correlation_heatmap.png)

---

## 🧠 Stage 2: Deep Neural Network (ANN) Architecture

```text
Input Layer (8 Clinical Features)
   │
   ▼
Dense (64 units) ──► BatchNormalization ──► ReLU ──► Dropout (0.3)
   │
   ▼
Dense (32 units) ──► BatchNormalization ──► ReLU ──► Dropout (0.3)
   │
   ▼
Output Layer: Dense (1 unit, Sigmoid Activation) ──► P(Diabetes)
```

| Hyperparameter | Configuration |
| :--- | :--- |
| **Optimizer** | `Adam (lr = 1e-3)` |
| **Loss Function** | `Binary Crossentropy` (Weighted by Class Balance) |
| **Batch Size / Max Epochs** | `512` / `100` |
| **Early Stopping** | `monitor='val_auc'`, `mode='max'`, `patience=10` |
| **LR Scheduler** | `ReduceLROnPlateau` (`factor=0.5`, `patience=5`, `min_lr=1e-6`) |

---

## 📈 Training Performance & Clinical Evaluation

### Training vs. Validation Trajectories (Loss, ROC-AUC, Recall)
![Training Curves](assets/training_curves.png)

### Confusion Matrix (Sensitivity vs. Specificity Breakdown)
![Confusion Matrix](assets/confusion_matrix.png)

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR-USERNAME/diabetes-risk-prediction-ann.git](https://github.com/YOUR-USERNAME/diabetes-risk-prediction-ann.git)
cd diabetes-risk-prediction-ann
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
1. Place `diabetes_prediction_dataset.csv` (from [Kaggle Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)) in the project root.
2. Run **`01_eda_and_preprocessing.ipynb`** to clean the data, generate EDA visualizations, and export `X_train.csv`, `X_val.csv`, `X_test.csv`, `y_train.csv`, `y_val.csv`, and `y_test.csv`.
3. Run **`02_ann_model_training.ipynb`** to train the TensorFlow/Keras ANN and generate the evaluation report.
