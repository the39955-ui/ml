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

from sklearn.tree import DecisionTreeClassifier, export_text

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
pred = model.predict(X_test)
cv = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=42), X, y,
                     cv=StratifiedKFold(5, shuffle=True, random_state=42))
print("Accuracy:", accuracy_score(y_test, pred), "5-fold CV:", cv.mean())
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("Precision:", precision_score(y_test, pred, average="macro"))
print("Recall:", recall_score(y_test, pred, average="macro"))
print("F1:", f1_score(y_test, pred, average="macro"))
print("Tree rules:\n", export_text(model, feature_names=list(X.columns)))
