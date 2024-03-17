import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# read model from file
model_filename = "ima_model.sav"
loaded_model = pickle.load(open(model_filename, "rb"))

# Load data
new_dataset_file = "data/ima_fem_new_cases.csv"
data = pd.read_csv(new_dataset_file)

# encode text features
le = LabelEncoder()
data["sex"] = le.fit_transform(data["sex"])

# Standardize the features
scaler = StandardScaler()
X_scale = scaler.fit_transform(data)

# get prediction
y_pred = loaded_model.predict(X_scale)
print(y_pred)
