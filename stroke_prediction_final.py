# Cell 1 — استيراد المكتبات المطلوبة للمشروع
# ✅ أضفنا: GridSearchCV, RandomizedSearchCV, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix, log_loss,
                              ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical

print("done!")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 — تحميل الداتا وعرضها
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print("Shape:", df.shape)
df.head()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 — معلومات عن الداتا (نوع كل عمود وعدد القيم)
df.info()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 — إحصاءات وصفية للداتا (mean, std, min, max)
df.describe()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 — التحقق من القيم الناقصة في كل عمود
df.isnull().sum()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 6 — توزيع الـ Target (stroke vs no stroke)
# الداتا Imbalanced — حالات الجلطة أقل بكتير
print(df['stroke'].value_counts())
print(f"\nStroke %: {df['stroke'].mean()*100:.1f}%")

df['stroke'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution')
plt.xticks([0, 1], ['No Stroke', 'Stroke'], rotation=0)
plt.ylabel('Count')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 7 — تحليل بصري للداتا
# 1- توزيع العمر للمصابين وغير المصابين
# 2- العلاقة بين الجلوكوز والـ BMI
# 3- Heatmap للـ Correlation بين الـ Features
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
df[df['stroke']==0]['age'].plot(kind='kde', color='green', label='No Stroke')
df[df['stroke']==1]['age'].plot(kind='kde', color='red', label='Stroke')
plt.title('Age Distribution')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(df[df['stroke']==0]['avg_glucose_level'],
            df[df['stroke']==0]['bmi'],
            alpha=0.3, s=6, color='green', label='No Stroke')
plt.scatter(df[df['stroke']==1]['avg_glucose_level'],
            df[df['stroke']==1]['bmi'],
            alpha=0.8, s=15, color='red', label='Stroke')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('Glucose vs BMI')
plt.legend()

plt.subplot(1, 3, 3)
num_cols = df.select_dtypes(include='number').drop('id', axis=1, errors='ignore')
sns.heatmap(num_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation')

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 8 — تأثير الضغط وأمراض القلب على الجلطة
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
pd.crosstab(df['hypertension'], df['stroke']).plot(
    kind='bar', color=['green', 'red'], ax=plt.gca())
plt.title('Hypertension vs Stroke')
plt.xticks([0, 1], ['No Hypertension', 'Hypertension'], rotation=0)
plt.legend(['No Stroke', 'Stroke'])

plt.subplot(1, 2, 2)
pd.crosstab(df['heart_disease'], df['stroke']).plot(
    kind='bar', color=['green', 'red'], ax=plt.gca())
plt.title('Heart Disease vs Stroke')
plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'], rotation=0)
plt.legend(['No Stroke', 'Stroke'])

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 9 — تنظيف الداتا
# 1- حذف عمود الـ ID (مش مفيد للموديل)
# 2- ملء القيم الناقصة في BMI بالـ Median
# 3- حذف الصف الوحيد اللي فيه gender = 'Other'
df = df.drop('id', axis=1)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df = df[df['gender'] != 'Other'].reset_index(drop=True)

print("Shape after cleaning:", df.shape)
df.isnull().sum()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 10 — تحويل الأعمدة النصية لأرقام بـ LabelEncoder
# عشان الموديل يقدر يتعامل معاها
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("Columns encoded:", cat_cols)
df.head()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 11 — تقسيم X و y وعمل Split وScaling
# stratify=y عشان نحافظ على نفس نسبة الـ stroke في Train و Test
X = df.drop('stroke', axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Scaling done!")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 12 — SMOTE لحل مشكلة الـ Imbalanced Data
# بيعمل بيانات اصطناعية لحالات الجلطة عشان يوازن الداتا
print("Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

import collections
print("\nAfter SMOTE:")
print(collections.Counter(y_train_res))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 13 — دالة لتقييم أي موديل وطباعة كل المقاييس
# Accuracy, F1-Score, AUC-ROC, Classification Report, Confusion Matrix
def evaluate(name, model, X_te, y_te):
    y_pred = model.predict(X_te)

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
    else:
        auc = 0

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, zero_division=0)

    print(f"===== {name} =====")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"F1-Score : {f1*100:.2f}%")
    print(f"AUC-ROC  : {auc*100:.2f}%")
    print()
    print(classification_report(y_te, y_pred,
          target_names=['No Stroke', 'Stroke'], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))
    print()

    return acc, f1, auc

# ─────────────────────────────────────────────────────────────────────────────
# Cell 14 — Linear Regression للمقارنة
# مش مناسبة للـ Classification عشان الناتج مش بين 0 و 1
lin = LinearRegression()
lin.fit(X_train_res, y_train_res)

y_raw  = lin.predict(X_test_scaled)
y_pred = (y_raw >= 0.5).astype(int)

print("Linear Regression")
print(f"Prediction range: {y_raw.min():.2f} to {y_raw.max():.2f}  ← goes outside [0,1]!")
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score : {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")
print()
print(classification_report(y_test, y_pred,
      target_names=['No Stroke', 'Stroke'], zero_division=0))

acc_lin = accuracy_score(y_test, y_pred)
f1_lin  = f1_score(y_test, y_pred, zero_division=0)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 15 — OLS (Normal Equation) لحساب الـ w يدوي
# w = (XᵀX)⁻¹ Xᵀy — نفس ما اتعلمنا في الـ Linear Regression
intercept_col = np.ones((X_train_res.shape[0], 1))
X1 = np.concatenate([intercept_col, X_train_res], axis=1)

z = np.linalg.inv(np.dot(X1.T, X1))
w = np.dot(z, np.dot(X1.T, y_train_res))

print("OLS Weights:")
print("intercept =", w[0])
for i in range(1, 4):
    print(f"coef{i}     =", w[i])

# ─────────────────────────────────────────────────────────────────────────────
# Cell 16 — Logistic Regression
# الأنسب للـ Binary Classification
log = LogisticRegression(max_iter=1000, random_state=42)
log.fit(X_train_res, y_train_res)
acc_log, f1_log, auc_log = evaluate("Logistic Regression", log, X_test_scaled, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 17 — LOG_LOSS للـ Logistic Regression (طريقة الدكتورة)
# LOG_LOSS بـ sklearn
y_pred_prob = log.predict_proba(X_test_scaled)
LOG_LOSS = log_loss(y_test, y_pred_prob)
LOG_LOSS

# LOG_LOSS يدوي بالمعادلة
y_test_cat = to_categorical(y_test)
loss = -np.sum(y_test_cat * np.log(y_pred_prob), axis=1).mean()
print("sklearn LOG_LOSS:", LOG_LOSS)
print("Manual  LOG_LOSS:", loss)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 18 — SVM بـ RBF Kernel
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_res, y_train_res)
acc_svm, f1_svm, auc_svm = evaluate("SVM (RBF)", svm, X_test_scaled, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 19 — Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_res, y_train_res)
acc_dt, f1_dt, auc_dt = evaluate("Decision Tree", dt, X_test_scaled, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 20 — Random Forest (أحسن موديل غالباً)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
acc_rf, f1_rf, auc_rf = evaluate("Random Forest", rf, X_test_scaled, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 21 — KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_res, y_train_res)
acc_knn, f1_knn, auc_knn = evaluate("KNN (k=5)", knn, X_test_scaled, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 22 — Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_res, y_train_res)
acc_gb, f1_gb, auc_gb = evaluate("Gradient Boosting", gb, X_test_scaled, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 23 — مقارنة كل الموديلات في جدول واحد
results = pd.DataFrame({
    'Model':    ['Linear Reg', 'Logistic Reg', 'SVM', 'Decision Tree',
                 'Random Forest', 'KNN', 'Gradient Boosting'],
    'Accuracy': [acc_lin, acc_log, acc_svm, acc_dt, acc_rf, acc_knn, acc_gb],
    'F1-Score': [f1_lin,  f1_log,  f1_svm,  f1_dt,  f1_rf,  f1_knn,  f1_gb],
    'AUC-ROC':  [0,       auc_log, auc_svm, auc_dt, auc_rf, auc_knn, auc_gb],
})

results = results.sort_values('AUC-ROC', ascending=False).reset_index(drop=True)

display_df = results.copy()
for col in ['Accuracy', 'F1-Score', 'AUC-ROC']:
    display_df[col] = display_df[col].map(lambda x: f'{x*100:.2f}%')

print(display_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 24 — رسم بياني لمقارنة الموديلات
models = results['Model']
x = np.arange(len(models))
w = 0.25

plt.figure(figsize=(13, 5))
plt.bar(x - w,  results['Accuracy']*100, w, label='Accuracy',  color='steelblue')
plt.bar(x,      results['F1-Score']*100, w, label='F1-Score',  color='orange')
plt.bar(x + w,  results['AUC-ROC']*100,  w, label='AUC-ROC',  color='green')
plt.xticks(x, models, rotation=30, ha='right')
plt.ylabel('%')
plt.ylim(0, 110)
plt.title('Model Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 25 — ✅ NEW: رسم الـ Confusion Matrix بصرياً بدل الـ print بس
# بيوضح إزاي الموديل بيصنف الـ 4 حالات:
# True Negative, False Positive, False Negative, True Positive
# اخترنا Logistic Regression و Random Forest لأنهم الأكثر شيوعاً للمقارنة
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Logistic Regression
cm_log = confusion_matrix(y_test, log.predict(X_test_scaled))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_log,
                               display_labels=['No Stroke', 'Stroke'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Logistic Regression')

# Random Forest
cm_rf = confusion_matrix(y_test, rf.predict(X_test_scaled))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_rf,
                                display_labels=['No Stroke', 'Stroke'])
disp2.plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title('Random Forest')

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 26 — ✅ NEW: Grid Search بيجرب كل الـ combinations الممكنة من الـ parameters
# وبيختار أحسنهم بناءً على الـ AUC-ROC
# بطيء لكن دقيق — لأنه بيجرب كل حاجة
# عدد الـ combinations هنا = 3 × 3 × 2 = 18 combination × 5 folds = 90 fit
param_grid = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [3, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,                    # 5-Fold Cross Validation
    scoring='roc_auc',       # الأهم في Medical Data
    n_jobs=-1,               # يستخدم كل الـ CPU cores
    verbose=1
)

grid_search.fit(X_train_res, y_train_res)

print("Best Parameters:", grid_search.best_params_)
print("Best AUC-ROC   :", round(grid_search.best_score_ * 100, 2), "%")

# نقيّم الـ Best Model
best_rf_grid = grid_search.best_estimator_
acc_g, f1_g, auc_g = evaluate("RF — Grid Search", best_rf_grid, X_test_scaled, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 27 — ✅ NEW: Randomized Search بيجرب عدد محدود من الـ combinations عشوائية
# أسرع من Grid Search — مناسب لو الـ parameters كتير
# عدد الـ combinations الكلي = 5×5×3×3 = 225 — بس بنجرب 20 بس منهم
param_dist = {
    'n_estimators':      [50, 100, 150, 200, 300],
    'max_depth':         [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,               # بيجرب 20 combination عشوائية بس
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train_res, y_train_res)

print("Best Parameters:", random_search.best_params_)
print("Best AUC-ROC   :", round(random_search.best_score_ * 100, 2), "%")

# نقيّم الـ Best Model
best_rf_random = random_search.best_estimator_
acc_r, f1_r, auc_r = evaluate("RF — Randomized Search", best_rf_random, X_test_scaled, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 28 — ✅ NEW: مقارنة الـ Random Forest العادي مع Grid وRandomized Search
# بيوضح إيه اللي اتحسن بعد الـ Hyperparameter Tuning
comparison = pd.DataFrame({
    'Model':    ['RF — Default', 'RF — Grid Search', 'RF — Randomized Search'],
    'Accuracy': [acc_rf,  acc_g,  acc_r],
    'F1-Score': [f1_rf,   f1_g,   f1_r],
    'AUC-ROC':  [auc_rf,  auc_g,  auc_r],
})

for col in ['Accuracy', 'F1-Score', 'AUC-ROC']:
    comparison[col] = comparison[col].map(lambda x: f'{x*100:.2f}%')

print(comparison.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 29 — Feature Importance من الـ Random Forest
# بتوضح أهم الـ Features في التنبؤ بالجلطة
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False)

print(feat_imp.round(4))

feat_imp.sort_values().plot(kind='barh', color='steelblue', figsize=(8, 5))
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 30 — دالة التنبؤ بخطر الجلطة لمريض جديد
# بتاخد بيانات المريض وبتقول هل عنده خطر جلطة أو لأ
def predict_stroke(age, hypertension, heart_disease, avg_glucose_level, bmi,
                   gender=1, ever_married=1, work_type=2,
                   Residence_type=1, smoking_status=1):

    patient = np.array([[gender, age, hypertension, heart_disease,
                         ever_married, work_type, Residence_type,
                         avg_glucose_level, bmi, smoking_status]])

    patient_scaled = scaler.transform(patient)
    pred = rf.predict(patient_scaled)[0]
    prob = rf.predict_proba(patient_scaled)[0][1]

    print(f"Prediction  : {'STROKE RISK ⚠️' if pred == 1 else 'No Stroke ✅'}")
    print(f"Probability : {prob*100:.1f}%")

    return pred, prob


print("=== High Risk Patient ===")
predict_stroke(age=75, hypertension=1, heart_disease=1,
               avg_glucose_level=220, bmi=33)

print("\n=== Low Risk Patient ===")
predict_stroke(age=25, hypertension=0, heart_disease=0,
               avg_glucose_level=88, bmi=21)

print("\n=== Medium Risk Patient ===")
predict_stroke(age=55, hypertension=1, heart_disease=0,
               avg_glucose_level=150, bmi=27)
