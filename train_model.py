import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

random_state = 52_889
n_estimators = 1_000
test_size = 0.2
n_splits = 20
model_filename = "ima_model.sav"
data_file = "data/ima_data_fem.csv"

# Load data
data = pd.read_csv(data_file)

# split features and target
y = data["recommended"]
X = data.drop(columns="recommended")

# encode text features
le = LabelEncoder()
X["sex"] = le.fit_transform(X["sex"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state,
)

# Perform cross-validation
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
cv_results = cross_val_score(
    rf_classifier, X_train, y_train, cv=kfold, scoring="accuracy"
)

# Output the mean and standard deviation of the cross-validation scores
print(
    f"CV Mean Accuracy: {cv_results.mean():.4f}, Standard Deviation: {cv_results.std():.4f}"
)
# Train the classifier on the training set
rf_classifier.fit(X_train, y_train)

# Predict the class for new observations
y_pred = rf_classifier.predict(X_test)

# Evaluate the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Default Accuracy on test set: {accuracy:.4f}")

# Save trained model to file
pickle.dump(rf_classifier, open(model_filename, "wb"))
