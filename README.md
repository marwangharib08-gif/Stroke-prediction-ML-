# 🧠 Stroke Prediction — ML Pipeline
### Binary Classification | Healthcare Dataset
**Marwan Gharib & Mohamed Abdelrehim | NTI Program 2026**

---

## 🎯 Problem
Stroke is the 2nd leading cause of death globally.  
This project builds an ML pipeline to predict stroke risk  
from patient health records using 7 classification models.

---

## 📊 Dataset
| Property | Value |
|---|---|
| Source | Kaggle — Healthcare Stroke Dataset |
| Records | 5,110 patients |
| Features | 10 clinical features |
| Class Balance | 95.1% No Stroke / 4.9% Stroke |

---

## 🤖 Models & Results
| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| **Logistic Regression** | 74.56% | 23.53% | **83.68% 🥇** |
| **Gradient Boosting** | 87.96% | **28.07% 🥇** | 80.37% |
| **Random Forest** | **93.05% 🥇** | 18.39% | 77.15% |
| SVM (RBF) | 75.44% | 19.29% | 76.42% |
| Decision Tree | 74.17% | 22.81% | 75.91% |
| KNN (k=5) | 79.94% | 14.94% | 66.36% |


---

## ⚙️ Pipeline
```
Raw Data → EDA → Handle Nulls → Label Encoding
→ Train/Test Split → StandardScaler → SMOTE
→ Train 7 Models → Hyperparameter Tuning
→ LOG LOSS Manual → Feature Importance 
→ Prediction Function
```

---

## 🔑 Key Findings
- **Age** is the #1 stroke predictor (38.1% importance)
- **Logistic Regression** wins on AUC (best for medical detection)
- **Random Forest** wins on Accuracy (used in final prediction)
- **SMOTE** balanced the dataset from 3,888:199 → 3,888:3,888
- Manual **LOG LOSS** match sklearn exactly ✅

---


## 🚀 How to Run
1. Open in Google Colab: [![Open In Colab](https://colab.research.google.com/drive/1Oo7xIj2TVaheYmQRcQtpLOPB1WJRQ36m?usp=sharing)
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and upload `healthcare-dataset-stroke-data.csv` to Colab
3. Run all cells

---

## 👥 Team
| Name | GitHub |
|---|---|
| Marwan Gharib | [@marwangharib08-gif](https://github.com/marwangharib08-gif) |
| Mohamed Abdelrehim | [@mh0748761-png](https://github.com/mh0748761-png) |
