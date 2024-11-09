# REPORT

# Part I
## Metrics
Not all metrics work for classification and for regression. The metrics that we chose are:
1. `Mean squared error`:
2. `mean absolute error`:
3. `R² score`:
4. `accuracy`:
5. `precision`:
6. `recall`:
7. `f1 score`: 

Accuracy, precision, recall and f1 score can be used as metrics for classification. 
R² score, mean squared error and mean absolute error can be used as metrics for regression. 

## Regression and classification models
We decided to use models from the scikit-learn library, because this made implementation a lot easier. Next to that,
it ensured that we would not make many implementation errors. Since this is quite a large project, it saved us 
quite some time not having to worry to much about the implementation and inner workings of each model. 

The models that we decided to use for classification were:
1. `logistic regression`: binary classification
2. `Decision Tree Classifier`: 
3. `Random forest Classifier`: 

The models that we decided to use for regression were:
1. `Linear regression`: 
2. `Decision Tree Regressor`: 
3. `Random forest Regressor`: 

A decision tree is ...
A random forest search is ... 

## Change to Dataset() read method:
There is a slight change to the read() method of Dataset class. This is, because when running the tests to see whether 
detect_feature_type() was working properly, we got a 'UnicodeDecodeError' error. Because of that, we tried to see if it would 
work if the bytes were read using a different encoding (`latin1` instead of `UTF-8`). If it would encounter lines with 
incorrect fields, they would be skipped. 

