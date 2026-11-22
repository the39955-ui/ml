# Machine Learning Practicals — Iris Dataset

This practical pack follows the uploaded **Machine Learning (DSC17/DSC-A5/GE7c)** guidelines. It includes all 12 prescribed experiments:
1. Naive Bayes classifier
2. Simple linear regression
3. Multiple linear regression
4. Polynomial regression
5. Lasso and Ridge regression
6. Logistic regression
7. Artificial Neural Network (MLP)
8. K-NN classifier
9. Decision tree classification
10. SVM classification
11. K-means clustering
12. Hierarchical clustering

## Dataset choice
All experiments use the classic Iris dataset. The included `data/iris.csv` has 150 observations, four numeric flower measurements, and the species label. Classification experiments predict species. Regression experiments predict petal length from one or more measurements. Clustering uses the measurements and does **not** train on species labels.

## Setup (Windows PowerShell)
```powershell
cd ML_Practicals_Iris
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run practicals
Run one experiment at a time from the project root:
```bash
python practicals/01_naive_bayes.py
python practicals/02_simple_linear_regression.py
# ... continue through 12
```
Or open the folder in VS Code and run the selected file. The scripts print metrics in the terminal. Hierarchical clustering additionally saves a dendrogram in `outputs/`.

## Evaluation used
- Regression: train/test split, 5-fold cross-validation, Mean Squared Error (MSE), and R².
- Classification: stratified train/test split, 5-fold stratified cross-validation, accuracy, confusion matrix, precision, recall, F1, error rate, and AUC where probability estimates are available. Specificity is reported for logistic regression using a macro one-vs-rest calculation.
- Clustering: inertia (K-means), silhouette score, and adjusted Rand index as an external comparison against the known species labels. Species labels are not used to fit clusters.

The confusion matrix convention is rows = true class and columns = predicted class. For multiclass classification, precision/recall/F1 are macro-averaged.

## Suggested practical record format
For each experiment, write:
1. Aim
2. Dataset and features used
3. Short theory / algorithm
4. Steps or procedure
5. Program
6. Output (paste terminal output)
7. Result / conclusion
8. Viva questions

## Important notes
- `random_state=42` makes the train/test split reproducible.
- Feature scaling is applied to distance- and gradient-sensitive models (KNN, SVM, MLP, Logistic Regression, Ridge/Lasso pipeline).
- Do not compare a clustering label number directly to a species name: cluster IDs are arbitrary. Use silhouette score or adjusted Rand index for evaluation.
- The syllabus asks for TP, TN, FP, FN. Each multiclass confusion matrix contains the information; for a one-vs-rest breakdown, calculate TP/TN/FP/FN separately for each class.
- These are runnable starter practicals. Add your own screenshots, observations, and handwritten explanations as required by your college.

## Syllabus coverage
The uploaded guidelines also mention preprocessing (feature scaling, feature selection, PCA), but the explicit practical-model list on page 2 contains the 12 experiments above. This ZIP covers that complete 12-item practical list. If your instructor expects separate preprocessing/PCA practicals, they can be added as an extension.
