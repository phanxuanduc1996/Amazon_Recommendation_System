# Amazon Recommendation System

## Compatibility
The code is tested using enviroment Spark 1.2.1 with Python 2.7 on Virtual Box.

## How to run

### Step1:
Start Hadoop on Virtual box.

### Step2:
Setup numpy, scipy library.

### Step3:
Run "export SPARK_HOME=/usr/hdp/2.2.4.2-2/spark" on terminal.

### Step4:
Move 4 file ``AmazonALS.py``, ``ratings.dat``, ``products.dat``, ``users.dat`` to folder of Hadoop:
	/workspace/AmazonRecommendation

### Step5:
Create folder in HDFS ``/user/AZ_P/input``, and move 3 file ``.dat`` to the newly created ``input`` directory with the following command:
	$ hdfs dfs -mkdir /user/AZ_P/input
	$ hdfs dfs �put <file_name> <path>

### Step6:
Go to the ``Demo`` folder containing the ``AmazonALS.py`` file. Open the terminal here and run:
	$ su root
	$ spark-submit AmazonALS.py <InputDirectory> <OutputFileName> <Iterations> <Partitions>
	$ spark-submit AmazonALS.py /user/AZ_P/input outputAmazonReco.dat 10 4

### Step7:
A ``outputAmazonReco.dat`` file has been created in the current directory with columns:
		userID, recommendedProduct, predictedRating
