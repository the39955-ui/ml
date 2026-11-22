from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "iris.csv"
df = pd.read_csv(DATA)
X = df.drop(columns="species")
y = df["species"]

from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
cv = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                     X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
cm = confusion_matrix(y_test, pred)
print("Accuracy:", accuracy_score(y_test, pred), "5-fold CV:", cv.mean())
print("Confusion matrix (rows=true, columns=predicted):\n", cm)
print("Precision:", precision_score(y_test, pred, average="macro"))
print("Recall:", recall_score(y_test, pred, average="macro"))
print("Specificity (macro one-vs-rest):",
      sum((cm.sum()-cm[i,:].sum()-cm[:,i].sum()+cm[i,i]) /
          (cm.sum()-cm[i,:].sum()) for i in range(len(cm))) / len(cm))
print("F1:", f1_score(y_test, pred, average="macro"))
print("Error rate:", 1-accuracy_score(y_test, pred))
print("AUC (OvR macro):", roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
