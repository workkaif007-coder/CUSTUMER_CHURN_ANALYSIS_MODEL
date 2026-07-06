import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import json

print("1. Loading dataset...")
df = pd.read_csv('dataset.csv')

# Ensure target variable is numeric (0 or 1)
df['Churn'] = df['Churn'].astype(int)

# Create Total Charges business KPI
df['Total Charges'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']

print("2. Clipping outliers...")
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if col == 'Churn':
        continue
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("3. Feature engineering...")
df['Total_Usage'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
df['Service_Stress'] = df['Customer service calls'] / (df['Account length'] + 1)
df['Revenue_Segment'] = pd.qcut(df['Total Charges'], q=3, labels=['Low', 'Medium', 'High'])

print("4. Encoding categorical features...")
le_intl = LabelEncoder()
df['International plan'] = le_intl.fit_transform(df['International plan'])

le_vmail = LabelEncoder()
df['Voice mail plan'] = le_vmail.fit_transform(df['Voice mail plan'])

le_churn = LabelEncoder()
df['Churn'] = le_churn.fit_transform(df['Churn'])

# Save states and revenue segments to reconstruct later
states_list = sorted(df['State'].unique().tolist())
joblib.dump(states_list, "states_list.pkl")

df = pd.get_dummies(df, columns=['State', 'Revenue_Segment'], drop_first=True, dtype=int)

print("5. Feature selection...")
corr_matrix = df.corr(numeric_only=True).abs()
high_corr = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr.add(corr_matrix.columns[i])

print(f"Dropping high correlation features: {high_corr}")
df.drop(columns=high_corr, inplace=True)
df.drop('Area code', axis=1, inplace=True, errors='ignore')

# Separate features and target
X = df.drop(['Churn'], axis=1)
y = df['Churn']

# Save the feature column list so the app can format inputs identically
model_features = X.columns.tolist()
with open("model_features.json", "w") as f:
    json.dump(model_features, f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_state=None, random_state=42) if False else train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("6. Hyperparameter tuning using Grid Search...")
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

print("7. Training final pipeline...")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", DecisionTreeClassifier(**grid_search.best_params_, random_state=42))
])
pipeline.fit(X_train, y_train)

# Save pipeline model
joblib.dump(pipeline, "best_model.pkl")
joblib.dump(pipeline, "churn_model.pkl")
print("Pipeline saved successfully as best_model.pkl and churn_model.pkl.")
