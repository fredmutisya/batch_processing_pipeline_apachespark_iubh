CREATE DATABASE mydatabase;
USE mydatabase;
CREATE TABLE mytable (
    id INT PRIMARY KEY,
    event_name VARCHAR(255),
    event_date DATE,
    event_location VARCHAR(255)
);



LOAD DATA INFILE '/event.csv'
INTO TABLE mytable
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n' 
IGNORE 1 LINES; 


SELECT * FROM mytable;
