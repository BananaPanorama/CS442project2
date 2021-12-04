import pyspark.sql.types as tp
from sklearn import metrics
from numpy import sqrt
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressionModel

spark = SparkSession.builder.appName('Wines').getOrCreate()

df = spark.read.format('csv').options(header='true', inferschema='true').load("test.csv",header=True)
features = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','pH','sulphates','alcohol'], outputCol = 'features')
v_df = features.transform(df)
v_df = v_df.select(['features','quality'])

#load the model 
print('Loading Linear Regression Model...')
lrModel = LinearRegressionModel.load('./lrModel')
rfModel = RandomForestRegressionModel.load('./rfModel')


#do linear regression and forestRegression
print('Performing Linear Regression on training data...')
predictions_linear = lrModel.transform(v_df)
predictions_forest = rfModel.transform(v_df)

#find linear regression results
y_true = predictions_linear.select("quality").toPandas()
y_pred = predictions_linear.select("prediction").toPandas()
r2 = metrics.r2_score(y_true, y_pred)
MAE = metrics.mean_absolute_error(y_true, y_pred)
MSE = metrics.mean_squared_error(y_true, y_pred)
rmse = sqrt(MSE)
f1 = metrics.f1_score(y_true, round(y_pred), average='macro')

#show linear regression results
print('Linear Regression Results')
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))
print('Mean Absolute Error = {0}'.format(MAE))
print('Mean Squared Error = {0}'.format(MSE))
print("Root Mean Squared Error (RMSE) = {0}".format(rmse))
print('r2_score = {0}'.format(r2))
print('f1_score = {0}'.format(f1))

#find forest regression results
y_true = predictions_forest.select("quality").toPandas()
y_pred = predictions_forest.select("prediction").toPandas()
r2 = metrics.r2_score(y_true, y_pred)
MAE = metrics.mean_absolute_error(y_true, y_pred)
MSE = metrics.mean_squared_error(y_true, y_pred)
rmse = sqrt(MSE)
f1 = metrics.f1_score(y_true, round(y_pred), average='macro')


#show forest regression results
print('Forest Regression Results')
print("Number of trees: " + str(rfModel.getNumTrees))
print('Mean Absolute Error = {0}'.format(MAE))
print('Mean Squared Error = {0}'.format(MSE))
print("Root Mean Squared Error (RMSE) = {0}".format(rmse))
print('r2_score = {0}'.format(r2))
print('f1_score = {0}'.format(f1))

predictions_linear.select("prediction","quality","features").show(10)
predictions_forest.select("prediction","quality","features").show(10)