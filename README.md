# Batch processing pipeline using Hadoop, Apache Spark and Docker- IUBH

# Data source

name: ecommerce.zip

Geography: Pakistan

Time period: 03/2016 – 08/2018

Source: https://www.kaggle.com/datasets/ertizaabbas/pakistan-ecommerce-dataset

Dataset: The dataset contains detailed information of e-commerce orders in Pakistan from March 2016 to August 2018. The file was editied by doubling the 500,000 original values to cross the 1,000,000 minimum threshold of the assignment and the date column was used to add a timestamp based on 1970 epoch. The dataset contains item details, shipping method, payment method like credit card, Easy-Paisa, Jazz-Cash, cash-on-delivery, product categories like fashion, mobile, electronics, appliance etc., date of order, SKU, price, quantity, total and customer ID. 

Size: 101 MB

File Type: CSV

# General structure

![Batch data processing pipeline](https://github.com/fredmutisya/batch_processing_pipeline_apachespark_iubh/assets/76942756/554bab6b-d8bc-46fd-82e8-c87311e58ed2)



# Docker images

# Project Docker Images

This project consists of six Docker images for hadoop, ambari, mysql,tensorflow and custom built images named spark_preprocessing_image and machine_learning_image. Follow the steps below to get started:

### Prerequisites

- Docker must be installed on your machine. You can download Docker from the official website: [https://www.docker.com/get-started](https://www.docker.com/get-started). Docker compose is also required.

### Accessing the Docker Images

### Accessing the Docker images using  Docker compose

This can be done by running the docker_compose_batch_processing.yml file to run all images in their defined sequence or use the individual docker compose yml files for each image

### Accessing the Docker images from the online repository- Dockerhub
I created 3 docker images and uploaded them on Docker hub under fredmutisya while i provided the images for hadoop and tensorflow from trusted repositories on dockerhub. The instructions for running the images are as follows:

1. Open a terminal or command prompt.

2. Pull the Docker images from Docker Hub using the `docker pull` command. Run the following commands for each image:

   ```shell
   docker pull rancher/hadoop-base
   docker pull hortonworks/ambari-server
   docker pull fredmutisya/mysql
   docker pull fredmutisya/spark_preprocessing_image
   docker pull fredmutisya/machine_learning_image
   docker pull tensorflow/tensorflow
   ```

3. Once the images are downloaded, you can verify their presence by running `docker images` or `docker image ls`. You should see the above images listed.

### Running the Docker Containers

After pulling the Docker images, you can run the corresponding Docker containers for each image. 



### Accessing the Services

Now that the Docker containers are running, you can access the services provided by each container:

- **Hadoop:** The Hadoop services are available within the `hadoop_container`. 

- **MySQL:** You can connect to the MySQL database by using the host `localhost`, port `3306`, and the credentials you set when running the MySQL container.

- **Spark Preprocessing:** Access the Spark preprocessing environment 

- **Machine Learning:** Access the machine learning service 

- **Tensorflow:** Access Tensorflow and Keras


For the management of the hadoop distributed file system, hive and apache spark, Apache Ambari can be used to manage the system. For deep learning Tensorflow and Keras can be utilized.

# Batch processing pipeline 

The batch processing pipeline involves multiple steps to process ecommerce data, store it in a MySQL database, transfer it to HDFS using Sqoop, convert it back to MySQL format, preprocess it using Apache Spark and PySpark, and perform machine learning tasks using Spark MLlib and scikit-learn. Here are the different steps in the pipeline:

1. **Data Ingestion**: Ecommerce data is retrieved from kaggle to mimic the output from a hadoop distributed file system

2. **MySQL Database Storage**: The raw ecommerce data is stored in a MySQL database to mimic data stored in SQl databases

3. **Data Transfer with Sqoop**: Sqoop is used to transfer the data from the MySQL database to HDFS, allowing for efficient processing and analysis with Hadoop.

4. **HDFS to MySQL Conversion**: Sqoop was utilized to extract the data from HDFS and convert it back to MySQL format for further analysis and integration.

5. **Preprocessing with Apache Spark and PySpark**: Apache Spark and PySpark are used to preprocess the data. This is is implemented in the file spark_preprocessing.pyin the following steps:

First, it imports the required dependencies, including pyspark and pandas. Then, a Spark session is initialized with specific configuration settings. The code defines the expected schema for the dataset using the StructType and StructField classes. Next, it loads the eCommerce dataset from a CSV file, applying the defined schema to create a DataFrame. Data type transformations are applied to specific columns using the withColumn and cast functions. Date columns ('created_at', 'Working Date', and 'Customer Since') are transformed from strings to the DateType using the to_date function. Additional transformations split the 'created_at' column into 'year', 'month', and 'day' columns using Spark SQL functions. Invalid rows with 'Year' values outside the range of 1900-2100 are filtered out. The Spark DataFrame is then converted to a Pandas DataFrame for further processing or analysis. Finally, the resulting Pandas DataFrame is exported to a CSV file named 'ml_output.csv' in overwrite mode, including headers and without an index.


6. **Machine Learning with Spark MLlib and scikitlearn**: Spark MLlib & scikitlean were utilized to perform feature selection and initial machine learning tasks on the preprocessed data. 

As a demonstration of the pipeline capabilities, only basic machine learning tasks are showcased. 
Firstly, the necessary dependencies are imported, including pandas, sklearn, xgboost, and matplotlib.pyplot. The pandas DataFrame obtained from Spark preprocessing is copied to a new DataFrame named df_ecommerce. Columns with a significant number of missing values are removed using the dropna function, followed by the removal of rows with any remaining missing values from the DataFrame.
Next, a subset of columns is selected for the classification task and stored in the column_list variable. The categorical feature "sku" is label encoded using the LabelEncoder from the sklearn.preprocessing module, and the encoded values are added as a new column named "sku_encoded" to the DataFrame.

The DataFrame is further subsetted to include only the columns specified in column_list. Categorical variables "category_name_1" and "payment_method" are one-hot encoded using the pd.get_dummies function, expanding them into multiple binary columns. The data is split into features (X) and the target variable (y). The dataset is then divided into training and testing sets using the train_test_split function from sklearn.model_selection.

A decision tree classifier is initialized, trained on the training data, and used to make predictions on the testing data. The accuracy of the decision tree model is calculated using the accuracy_score function from sklearn.metrics. Additionally, a decision tree classifier with a maximum depth of 3 is initialized, fitted to the training data, and plotted using the plot_tree function from sklearn.tree and matplotlib.pyplot. A random forest classifier is also initialized, trained on the training data, and used to make predictions on the testing data. The accuracy of the random forest model is calculated using accuracy_score. Furthermore, an XGBoost classifier is initialized, trained on the training data, and used to make predictions on the testing data. The accuracy of the XGBoost model is calculated using accuracy_score.

The preprocessing has a csv output which can be fed into a deep learning module using the tensorflow image which has intergrated Keras.

For managing the HDFS system in practice, it is recommend to use Hortonworks/Cloudera which has Apache Ambari, Sqoop, Apache spark, HIVE and HDFS in the same installation





