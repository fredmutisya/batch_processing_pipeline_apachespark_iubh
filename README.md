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


# Docker images

# Project Docker Images

This project consists of four Docker images: `mysql`, `machine_learning_image`, `spark_preprocessing_image`, and `hadoop_image`. This README file provides instructions on how to access and use these Docker images for the project. Follow the steps below to get started:

### Prerequisites

- Docker must be installed on your machine. You can download Docker from the official website: [https://www.docker.com/get-started](https://www.docker.com/get-started)

### Accessing the Docker Images

1. Open a terminal or command prompt.

2. Pull the Docker images from Docker Hub using the `docker pull` command. Run the following commands for each image:

   ```shell
   docker pull fredmutisya/mysql
   docker pull fredmutisya/machine_learning_image
   docker pull fredmutisya/spark_preprocessing_image
   docker pull rancher/hadoop-base
   ```

3. Once the images are downloaded, you can verify their presence by running `docker images`. You should see the four images listed: `mysql`, `machine_learning_image`, `spark_preprocessing_image`, and `hadoop_image`.

### Running the Docker Containers

Now that you have pulled the Docker images, you can run the corresponding Docker containers for each image. Follow the steps below:

#### MySQL Container

1. Run the MySQL container using the following command:

   ```shell
   docker run -d --name mysql_container -p 3306:3306 -e MYSQL_ROOT_PASSWORD=<password> fredmutisya/mysql
   ```

   Replace `<password>` with your desired password for the MySQL root user.

2. Wait for the container to start. You can check the logs using `docker logs mysql_container` to verify that the MySQL server is running.

#### Machine Learning Container

1. Run the machine learning container using the following command:

   ```shell
   docker run -d --name ml_container --link mysql_container -p 5000:5000 fredmutisya/machine_learning_image
   ```

   The `--link` flag is used to connect the machine learning container to the MySQL container.

2. Wait for the container to start. You can check the logs using `docker logs ml_container` to verify that the machine learning server is running.

#### Spark Preprocessing Container

1. Run the Spark preprocessing container using the following command:

   ```shell
   docker run -d --name spark_container -p 8888:8888 fredmutisya/spark_preprocessing_image
   ```

2. Wait for the container to start. You can check the logs using `docker logs spark_container` to verify that the Spark preprocessing environment is running.

#### Hadoop Container

1. Run the Hadoop container using the following command:

   ```shell
   docker run -d --name hadoop_container fredmutisya/hadoop_image
   ```

2. Wait for the container to start. You can check the logs using `docker logs hadoop_container` to verify that the Hadoop services are running.

### Accessing the Services

Now that the Docker containers are running, you can access the services provided by each container:

- **MySQL:** You can connect to the MySQL database by using the host `localhost`, port `3306`, and the credentials you set when running the MySQL container.

- **Machine Learning:** Access the machine learning service by using the URL `http://localhost:5000` in your web browser.

- **Spark Preprocessing:** Access the Spark preprocessing environment by using the URL `http://localhost:8888` in your web browser.

- **Hadoop:** The Hadoop services are now available within the `hadoop_container`. 

## For the management of the hadoop distributed file system, hive and apache spark, Apache Ambari can be used to manage the system.

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



