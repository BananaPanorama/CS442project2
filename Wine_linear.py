from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


spark = SparkSession.builder.appName('Wines').getOrCreate()

df = spark.read.format('csv').options(header='true', inferschema='true').load("train.csv",header=True)

#features = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'], outputCol = 'features')
features = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','pH','sulphates','alcohol'], outputCol = 'features')
v_df = features.transform(df)
v_df = v_df.select(['features','quality'])

# Fit the model
lr = LinearRegression(featuresCol = 'features', labelCol = 'quality', maxIter = 25000, regParam = 0.05, elasticNetParam = 0.08, tol = 1e-07)
lrModel = lr.fit(v_df)
lrModel.write().overwrite().save("./lrModel")
