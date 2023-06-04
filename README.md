# Batch processing pipeline using Hadoop, Apache Spark and Docker- IUBH

# Data source

name: ecommerce.zip

Geography: Pakistan

Time period: 03/2016 – 08/2018

Source: https://www.kaggle.com/datasets/ertizaabbas/pakistan-ecommerce-dataset

Dataset: The dataset contains detailed information of half a million e-commerce orders in Pakistan from March 2016 to August 2018. It contains item details, shipping method, payment method like credit card, Easy-Paisa, Jazz-Cash, cash-on-delivery, product categories like fashion, mobile, electronics, appliance etc., date of order, SKU, price, quantity, total and customer ID. This is the most detailed dataset about e-commerce in Pakistan that you can find in the Public domain.

Variables: The dataset contains Item ID, Order Status (Completed, Cancelled, Refund), Date of Order, SKU, Price, Quantity, Grand Total, Category, Payment Method and Customer ID.

Size: 101 MB

File Type: CSV


# Batch processing pipeline 

The batch processing pipeline involves multiple steps to process ecommerce data, store it in a MySQL database, transfer it to HDFS using Sqoop, convert it back to MySQL format, preprocess it using Apache Spark and PySpark, and perform machine learning tasks using Spark MLlib and scikit-learn. Here are the different steps in the pipeline:

1. **Data Ingestion**: Ecommerce data is retrieved from kaggle to mimic the output from a hadoop distributed file system

2. **MySQL Database Storage**: The raw ecommerce data is stored in a MySQL database to mimic data stored in SQl databases

3. **Data Transfer with Sqoop**: Sqoop is used to transfer the data from the MySQL database to HDFS, allowing for efficient processing and analysis with Hadoop.

4. **HDFS to MySQL Conversion**: Sqoop was utilized to extract the data from HDFS and convert it back to MySQL format for further analysis and integration.

5. **Preprocessing with Apache Spark and PySpark**: Apache Spark and PySpark are used to preprocess the data. This is is implemented in the file spark_preprocessing.pyin the following steps:

Importing Dependencies: The required libraries, such as pyspark, pandas, and specific modules from pyspark.sql, are imported.
Initializing Spark Session: The code initializes a Spark session with the application name 'eCommerce' and specific configuration settings.
Defining Schema: The expected schema for the eCommerce dataset is defined using the StructType and StructField classes.
Loading the Dataset: The code loads the eCommerce dataset from a CSV file, applying the defined schema to the DataFrame.
Data Type Transformation: The code converts specific columns to their proper data types using the withColumn and cast functions.
Date Transformation: The 'created_at', 'Working Date', and 'Customer Since' columns are transformed from strings to the DateType using the to_date function.
Additional Data Transformation: The 'created_at' column is split into 'year', 'month', and 'day' columns using Spark SQL functions.
Filtering Invalid Rows: Rows with 'Year' values outside the range of 1900-2100 are filtered out using the filter function.
Converting to Pandas DataFrame: The Spark DataFrame is converted to a Pandas DataFrame for further processing or analysis.
Exporting to CSV: The Pandas DataFrame is written to a CSV file named 'ml_output.csv' with headers, no index, and overwrite mode.



6. **Machine Learning with Spark MLlib and scikitlearn**: Spark MLlib & scikitlean were utilized to perform feature selection and initial machine learning tasks on the preprocessed data. 

Importing Dependencies: The required libraries, such as pandas, sklearn, xgboost, and matplotlib.pyplot, are imported.
Copying the DataFrame: The pandas DataFrame pandas_df obtained from Spark preprocessing is copied to df_ecommerce.
Removing Columns with Missing Values: Columns with more than 50% missing values are removed from the DataFrame using the dropna function.
Dropping Rows with Missing Values: Rows with any remaining missing values are dropped from the DataFrame.
Filtering Columns: A subset of columns to be used in the classification task is selected and stored in column_list.
Label Encoding: The categorical feature "sku" is encoded using LabelEncoder from sklearn.preprocessing, and the encoded values are stored in a new column called "sku_encoded".
Subset the DataFrame: The DataFrame is further subsetted to include only the columns specified in column_list.
One-Hot Encoding: The categorical variables "category_name_1" and "payment_method" are one-hot encoded using pd.get_dummies.
Splitting the Data: The DataFrame is split into features (X) and the target variable (y). The data is further split into training and testing sets using train_test_split from sklearn.model_selection.
Decision Tree Classifier: A decision tree classifier is initialized, trained on the training data, and used to make predictions on the testing data. The accuracy of the model is calculated using accuracy_score from sklearn.metrics.
Plotting the Decision Tree: A decision tree classifier is initialized with a maximum depth of 3, fitted to the training data, and plotted using plot_tree from sklearn.tree and matplotlib.pyplot.
Random Forest Classifier: A random forest classifier is initialized, trained on the training data, and used to make predictions on the testing data. The accuracy of the model is calculated using accuracy_score.
XGBoost Classifier: An XGBoost classifier is initialized, trained on the training data, and used to make predictions on the testing data. The accuracy of the model is calculated using accuracy_score.



