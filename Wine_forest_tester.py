from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName('Wines').getOrCreate()

df = spark.read.format('csv').options(header='true', inferschema='true').load("winequality-white.csv",header=True)


#features = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'], outputCol = 'features')
features = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','pH','sulphates','alcohol'], outputCol = 'features')
v_df = features.transform(df)
v_df = v_df.select(['features','quality'])
#v_df.show(3)

(train_df, test_df) = v_df.randomSplit([0.8, 0.2])

# Fit the model
rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'quality',numTrees = 100, maxDepth = 20, maxBins = 200, minInstancesPerNode = 1)
#rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'quality', numTrees = 100, maxDepth = 20, maxBins = 100, minInstancesPerNode = 2)
rfModel = rf.fit(train_df)



predictions = rfModel.transform(test_df)

predictions.select("prediction","quality","features").show(10)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="quality",predictionCol="prediction",metricName="rmse")

rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) = %g" % rmse)

y_true = predictions.select("quality").toPandas()
y_pred = predictions.select("prediction").toPandas()

r2 = r2_score(y_true, y_pred)
f1 = f1_score(y_true, round(y_pred), average='macro')
print('r2_score: {0}'.format(r2))
print('f1_score: {0}'.format(f1))