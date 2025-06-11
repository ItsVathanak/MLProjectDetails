#Step 1: Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#Step 2: Load datasets

df = pd.read_csv('data/food_ingredients_and_allergens.csv') #please copy the relative path of the csv file and paste here

#Display first few rows

print("First few rows of dataset:")
print(df.head())

#Step 3: Data Summary

print("\nSummary of dataset:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nNumber of samples:", len(df))

#Step 4: Data Visualization

#Class Distribution
sns.countplot(data=df, x='Prediction')
plt.title("Class Distribution: 'Contains' vs 'Does Not Contain'")
plt.show()

#Step 5: Data Preprocessing
#We are dropping 'Allergens' and 'Food Product', since they ar enot predictive or leak output
df = df.drop(columns=['Allergens','Food Product'])

# #Drop missing values
# df = df.dropna()

#Drop rows where 'Prediction" is missing
df.dropna(subset=['Prediction'], inplace=True)
print(f"\nDataframe shape after drpping rows that are missing 'Prediction': {df.shape}")

#Separate features and target
X = df.drop(columns=['Prediction'])
y = df['Prediction']

#One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# More visualization

# Heatmap of feature correlations (after encoding)
plt.figure(figsize=(10, 8))
correlation_matrix = X_encoded.corr()
sns.heatmap(correlation_matrix, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Encoded Features")
plt.show()

#Continue preprocessing
#Standardize using z-score
X_standardized = X_encoded.apply(zscore)
print("\nStandardized features (first five rows):")
print(X_standardized.head())

#Step 6: Split the dataset

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42, stratify=y)

#Because the dataset is imbalanced, we gotta use SMOTE to fix it
# Import SMOTE
from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nShape of X_train before SMOTE: {X_train.shape}")
print(f"Shape of X_train after SMOTE: {X_train_resampled.shape}")
print("\nClass distribution in y_train before SMOTE:")
print(y_train.value_counts())
print("\nClass distribution in y_train after SMOTE:")
print(y_train_resampled.value_counts())

#Step 7 & 8: Choosing ML algorithm and training
# We are going to try Logistic regression, random forest, naive bayes, and support vector machine

#import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Dictionary to hold models and their names
models = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Naive Bayes": GaussianNB(),  
    "SVM": SVC(kernel='linear', probability=True, class_weight='balanced')
}

# Evaluate each model
print("<==========Model evaluation section==========>")
model_predictions = {}
for name, model in models.items():
    print(f"\n===== {name} =====")
    
    # Train the model
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    #storing model for reuse
    model_predictions[name] = y_pred

    # Evaluation
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

print("<==========End of model evaluation section==========>")

print("\n<==========Performance comparison section==========>")
#Performance comparison using Accuracy and f1-score
from sklearn.metrics import accuracy_score, f1_score

# Dictionaries to store performance metrics
accuracy_scores = {}
f1_scores = {}

for name, model in models.items():
    print(f"\n===== {name} =====")

    #Get y_pred
    y_pred = model_predictions[name]

    # Scores
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='Contains',zero_division=0)

    # Store scores
    accuracy_scores[name] = acc
    f1_scores[name] = f1

    # Print evaluation
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    # print("Classification Report:\n", classification_report(y_test, y_pred,zero_division=0))

    # Save model (optional)
    #joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')

# Visualization
# Accuracy Plot
plt.figure(figsize=(10, 5))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.show()

# F1 Score Plot
plt.figure(figsize=(10, 5))
plt.bar(f1_scores.keys(), f1_scores.values(), color='salmon')
plt.title('Model F1 Score Comparison')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.show()

# 9. Inferencing (testing it on the remaining 20% of data)
final_model_name = ""
init_int = 0.0000
for name,value in f1_scores.items():
  
    if value >= init_int:
        init_int = value
        final_model_name = name

print(f"Model selected for highest f1 score: {final_model_name}")

#Test on firsst five samples
# print("\nSample Predictions (First five):")
# print("Predicted:", models[final_model_name].predict(X_test[:5]))
# print("Actual   :", y_test[:5].values)

#Test on random samples
# import random

# random_samples = X_test.sample(n=5, random_state=42)
# random_indices = random_samples.index

# #Get true labels for the samples
# true_labels = y_test.loc[random_indices]

# print("\nSample Predictions (Random):")
# print("Predicted:", models[final_model_name].predict(random_samples))
# print("Actual   :", true_labels.values)


# Filter test set for 'Does not contain' class
negative_class_indices = y_test[y_test == 'Does not contain'].index
X_negative = X_test.loc[negative_class_indices]
y_negative = y_test.loc[negative_class_indices]

# Randomly sample 5 rows (or fewer if not enough available)
sampled_X = X_negative.sample(n=min(5, len(X_negative)), random_state=42)
sampled_y = y_negative.loc[sampled_X.index]

# Show predictions
print("\nSample Predictions from 'Does not contain' class:")
print("Predicted:", models[final_model_name].predict(sampled_X))
print("Actual   :", sampled_y.values)
