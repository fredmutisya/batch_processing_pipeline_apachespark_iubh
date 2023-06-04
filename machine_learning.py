
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from spark_preprocessing import pandas_df

import xgboost as xgb


from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt


#Copying the pandas dataframe from the spark preproccessing


df_ecommerce = pandas_df.copy()

#Removing columns with more than 50% nas
threshold = len(df_ecommerce) * 0.5  # 50% threshold
df_ecommerce = df_ecommerce.dropna(axis=1, thresh=threshold)

#Droping rows with nas and then checking if there are any remaining nas
df_ecommerce = df_ecommerce.dropna(axis=0)

columns_with_nas = df_ecommerce.columns[df_ecommerce.isna().any()].tolist()
print(columns_with_nas)

#Filtering columns
column_list = ['status', 'price', 'qty_ordered', 'sku_encoded', 'grand_total', 'category_name_1', 'discount_amount', 'payment_method', 'Year', 'Month']


# Initialize LabelEncoder for sku feature
label_encoder = LabelEncoder()
# Encode the "sku" column
df_ecommerce['sku_encoded'] = label_encoder.fit_transform(df_ecommerce['sku'])

#Subset the dataframe
df_ecommerce = df_ecommerce[column_list]
df_ecommerce

#Convert the categorical variables
df_ecommerce = pd.get_dummies(df_ecommerce, columns=['category_name_1'])

df_ecommerce = pd.get_dummies(df_ecommerce, columns=['payment_method'])

# Create a new DataFrame to avoid the SettingWithCopyWarning
df_ecommerce_encoded = df_ecommerce.copy()
# Define features X and target y
X = df_ecommerce.drop('status', axis=1)
y = df_ecommerce['status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train


#Decision Tree Classifier
# Initialize DecisionTreeClassifier
clf_dt = DecisionTreeClassifier()
# Encode the target variable y_train
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
# Train the classifier
clf_dt.fit(X_train, y_train_encoded)
# Make predictions
y_pred_encoded = clf_dt.predict(X_test)
# Decode the predictions
y_pred = label_encoder.inverse_transform(y_pred_encoded)
# Check the accuracy of the model
print('Accuracy decision tree classifier:', accuracy_score(y_test, y_pred))



#Plot 
# Initialize DecisionTreeClassifier with max_depth
clf_dt = DecisionTreeClassifier(max_depth=3)
# Fit the classifier to your data
clf_dt.fit(X_train, y_train)
# Plot the decision tree with max_depth
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(clf_dt, feature_names=X_train.columns, class_names=clf_dt.classes_, filled=True, ax=ax)
# Display the decision tree plot
plt.show()



#Random forest
# Initialize RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
# Train the classifier
clf.fit(X_train, y_train)
# Make predictions
y_pred = clf.predict(X_test)
# Check the accuracy of the model
print('Accuracy random forest classifiers:', accuracy_score(y_test, y_pred))



#Xgboost
# Initialize XGBClassifier
clf_xg = xgb.XGBClassifier(n_estimators=100)
# Encode the target variable y_train
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
# Train the classifier
clf_xg.fit(X_train, y_train_encoded)
# Make predictions
y_pred_encoded = clf_xg.predict(X_test)
# Decode the predictions
y_pred = label_encoder.inverse_transform(y_pred_encoded)
# Check the accuracy of the model
print('Accuracy Xgboost:', accuracy_score(y_test, y_pred))
