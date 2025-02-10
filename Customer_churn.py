#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install shap')


# #### Import necessary libraries

# In[3]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
import shap


# #### Load Dataset
# 

# In[16]:


# Load dataset
file_path = "C:/Users/ayodi/Documents/RotorgapLocal/Customer_churn/Churn_dataset.csv"
df = pd.read_csv(file_path)
df.head()


# ##### Data Cleaning
# 

# In[12]:


# Drop irrelevant features
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
df.head()


# In[14]:


# Handling categorical feature encoding
data['Card Type'] = LabelEncoder().fit_transform(data['Card Type'])

# Identify categorical and continuous features
categorical_features = ['Geography', 'Gender']
continuous_features = [col for col in data.columns if col not in categorical_features + ['Exited']]


# #### One-Hot Encoding for categorical features

# In[18]:


# One-Hot for categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(df[categorical_features])
encoded_columns = encoder.get_feature_names_out(categorical_features)
X_encoded_df = pd.DataFrame(X_encoded, index=data.index, columns=encoded_columns)


# In[19]:


# Combine with continuous features
X_final = pd.concat([data[continuous_features].reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)
y = data['Exited']


# In[21]:


# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)
X_scaled_df = pd.DataFrame(X_scaled, index=df.index, columns=X_final.columns)


# In[22]:


# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)


# In[23]:


# Handling Imbalanced Data using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# In[24]:


# Define Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier()
}


# In[25]:


# Train and Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    results[name] = {'Accuracy': accuracy, 'AUC-ROC': auc_score}
    print(f"{name} Accuracy: {accuracy:.4f}, AUC-ROC: {auc_score}")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {name}")
    plt.show()


# In[26]:


# Model Comparison
results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='Accuracy', ascending=False))


# In[35]:


# # Feature Importance using SHAP
# rf = models['Random Forest']
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, show=False)
# plt.savefig('shap_summary_plot.png', bbox_inches='tight')
# shap.summary_plot(shap_values, X_test)
rf = models['Random Forest']
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap_values = np.array(shap_values)  # Ensure correct format
plt.figure()
shap.summary_plot(shap_values, X_test, max_display=10, show=False)
plt.savefig('shap_summary_plot.png', bbox_inches='tight')
plt.show()


# In[ ]:




