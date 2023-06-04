# Use a base image with Java and Hadoop dependencies
FROM openjdk:8-jdk

# Set environment variables
ENV HADOOP_VERSION=3.3.1
ENV HADOOP_HOME=/opt/hadoop

# Download and extract Hadoop
RUN curl -LO https://downloads.apache.org/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz \
    && tar -xvzf hadoop-$HADOOP_VERSION.tar.gz \
    && mv hadoop-$HADOOP_VERSION $HADOOP_HOME \
    && rm hadoop-$HADOOP_VERSION.tar.gz

# Set Hadoop configuration
ENV PATH=$HADOOP_HOME/bin:$PATH
COPY core-site.xml $HADOOP_HOME/etc/hadoop/
COPY hdfs-site.xml $HADOOP_HOME/etc/hadoop/
COPY mapred-site.xml $HADOOP_HOME/etc/hadoop/
COPY yarn-site.xml $HADOOP_HOME/etc/hadoop/

# Expose Hadoop ports
EXPOSE 50070 8088

# Start Hadoop services
CMD ["hadoop", "namenode", "-format"]
CMD ["hadoop", "daemons"]
CMD ["yarn", "resourcemanager"]



