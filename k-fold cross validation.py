import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

all_preds = []
all_true = []

for train_idx, test_idx in skf.split(X_combined, labels):
    X_train, X_test = X_combined[train_idx], X_combined[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    all_preds.extend(y_pred)
    all_true.extend(y_test)
