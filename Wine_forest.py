from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

spark = SparkSession.builder.appName('Wines').getOrCreate()

df = spark.read.format('csv').options(header='true', inferschema='true').load("train.csv",header=True)

features = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','pH','sulphates','alcohol'], outputCol = 'features')
v_df = features.transform(df)
v_df = v_df.select(['features','quality'])

# Fit the model
rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'quality',numTrees = 100, maxDepth = 20, maxBins = 200, minInstancesPerNode = 1)
rfModel = rf.fit(v_df)
rfModel.write().overwrite().save('./rfModel')
