# Predictive-Modeling-for-Agriculture
In this project, I developed multi-class classification models to optimize crop selection based on soil health. By analyzing Nitrogen, Phosphorous, Potassium, and pH levels, I identified the most critical features for yield. This data-driven approach helps farmers prioritize soil tests, ensuring cost-effective and sustainable outcomes.

# Import the required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# READ THE DATA INTO PANDAS DATAFRAME AND PERFORM EDA
# Load the dataset
crops = pd.read_csv("soil_measures.csv")
print(crops.head())

# Check for missing values and crop types
crops.isna().sum().sort_values()
crops['crop'].unique()

# SPLIT THE DATA
# Create the features and target variables
X = crops.drop('crop', axis = 1).values
y = crops['crop'].values
print(type(X), type(y))

# Split the data
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.3, stratify=y, random_state=21)

# EVALUATE FEATURE PERFORMANCE

# Make prediction from the feature Nitrogen
N_crop = crops[['N']].values
reg = LogisticRegression()
reg.fit(N_crop, y)
predict_N_crop = reg.score(N_crop, y)
print(predict_N_crop)

# Make prediction from the feature Phosphorus
P_crop = crops[['P']].values
reg = LogisticRegression()
reg.fit(P_crop, y)
predict_P_crop = reg.score(P_crop, y)
print(predict_P_crop)

# Make prediction from the feature Potassium
K_crop = crops[['K']].values
reg = LogisticRegression()
reg.fit(K_crop, y)
predict_K_crop = reg.score(K_crop, y)
print(predict_K_crop)

# Make prediction from the feature ph
ph_crop = crops[['ph']].values
reg = LogisticRegression()
reg.fit(ph_crop, y)
predict_ph_crop = reg.score(ph_crop, y)
print(predict_ph_crop)

# Create a dictionary to store each features predictive performance
feature_predictive_performance = {
    'Nitrogen': predict_N_crop,
    'Phosphorus': predict_P_crop,
    'Potassium': predict_K_crop,
    'ph': predict_ph_crop
}

# Loop through the features
for feature in feature_predictive_performance:
    print(f"{feature}: {feature_predictive_performance[feature]}")

# Train a multi-class classifier algorithm
reg.fit(X_train, y_train)

# Predict target values using the test set
y_pred = reg.predict(X_test)

# Evaluate the performance of each feature
y_pred_probs = reg.predict_proba(X_test)[:, 1]
print(y_pred_probs[0])

# CREATE THE BEST PREDICTIVE FEATURE VARIABLE
best_predictive_feature = {'K': predict_K_crop}
print(f"Best predictive feature: {best_predictive_feature}")
