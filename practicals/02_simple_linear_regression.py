from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "iris.csv")
# Predict petal length using only sepal length (one input feature).
X = df[["sepal length (cm)"]]
y = df["petal length (cm)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
cv = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
print("Slope:", model.coef_[0], "Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, pred), "R2:", r2_score(y_test, pred))
print("5-fold CV R2:", cv.mean())
print("Sample prediction:", model.predict([[5.1]])[0])
