import pyspark.sql.types as tp
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator




#def transData(data):
 #   return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

spark = SparkSession.builder.appName('Wines').getOrCreate()
sc = spark.sparkContext

#define schema
my_schema = tp.StructType([
    tp.StructField(name = 'fixed acidity', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'volatile acidity', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'citric acid', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'residual sugar', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'chlorides', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'free sulfur dioxide', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'total sulfur dioxide', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'density', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'pH', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'sulphates', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'alcohol', dataType = tp.DoubleType(), nullable = True),
    tp.StructField(name = 'quality', dataType = tp.IntegerType(), nullable = True)
])

df = spark.read.format('csv').options(header='true', inferschema='true').load("winequality-white.csv",header=True)
#df.show(5,True)
#df.printSchema()

#v_df = transData(df)
#v_df.show(5)

#featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(v_df)

#data = featureIndexer.transform(v_df)
#data.show(5,True)


#(train_df, test_df) = data.randomSplit([0.8, 0.2])
#train_df.show(5)

#lr = LinearRegression()

#pipeline = Pipeline(stages = [featureIndexer, lr])
#model = pipeline.fit(train_df)

# Print the coefficients and intercept for linear regression
#print("Coefficients: " + str(model.stages[-1].coefficients))
#print("Intercept: " + str(model.stages[-1].intercept))




#features = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'], outputCol = 'features')
features = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','pH','sulphates','alcohol'], outputCol = 'features')
v_df = features.transform(df)
v_df = v_df.select(['features','quality'])


(train_df, test_df) = v_df.randomSplit([0.8, 0.2])


# Fit the model
lr = LinearRegression(featuresCol = 'features', labelCol = 'quality', maxIter = 25000, regParam = 0.05, elasticNetParam = 0.08, tol = 1e-07)

lrModel = lr.fit(train_df)

# Print the coefficients and intercept for linear regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

predictions = lrModel.transform(test_df)

predictions.select("prediction","quality","features").show(10)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="quality",predictionCol="prediction",metricName="rmse")

rmse = evaluator.evaluate(predictions)


y_true = predictions.select("quality").toPandas()
y_pred = predictions.select("prediction").toPandas()

y_pred = round(y_pred)

r2 = r2_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred, average='macro')
print('r2_score: {0}'.format(r2))
print("Root Mean Squared Error (RMSE) = %g" % rmse)

print('f1_score: {0}'.format(f1))

lrModel.save("./lrModel")