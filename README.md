# ML_Tutorial
Sample Code (Python) and Data for Studying Machine Learning Algorithms 

## Prerequisite 
1.Required Library: Numpy, Matplot, Scipy, Tensorflow 

2.Knowledge of following topics 
- Linear algebra 
- Diffential formula (Matrices, Composite Function)
- Gradient Descent 
- Vectorized Implementation with Numpy 

Please refer to http://

## 1. Linear Regression
### Input Data

* [lr_data1.txt] - Single Feature Example (X, Y) from Andrew Ng's ML class
* [lr_data2_multi.txt] - Multiple Feature Example (X1, X2, Y) from Andrew Ng's ML class
* [lr_data3.txt] - Single Freture Example. X0 is included in data file(1, X1, Y). from 'Machine Learning in Action' 

### Code
* [_test_linear_regression*.py] - Test Code for Linear Regression with input data
* [linear_regression.py] - Functions for Linear regression 
* [ml_utils.py] - Utility functions (i.e., featureNormalize(X))

## 2. Logistic Regression
### Input Data

* [logistic_data1.txt] - Example from Andrew Ng's ML class (X1, X2, Y)
* [logistic_data2_reg.txt] - Example form Andrew Ng's ML class (+Polynomial and Regularization)
* [logistic_data3.txt] - Example from 'Machine Learning in Action' (X1, X2, Y) 

### Code
* [_test_logistic_regression*.py] - Test Code for Logistic Regression with input data
* [logistic_regression.py] - Functions for Linear regression 
* [ml_utils.py] - Utility functions (i.e., mapFeature(X): Polynomial Extension)

## 3. Softmax Regression
### Input Data
* [mnist/*] - data file from MNIST
 
### Code
* [_test_softmax.py] - Test code for Softmax Regression with Mnist Data
* [softmax.py] - Functions for Softmax Regression (Ref. UFLDL Tutorial)
* [data_utils.py] - Utility Function for data handlng (load MNIST data)
* [ml_utils.py] - Utility Function (onehotTransform(X)): Categoricla Data to one-hot)  


## Reference
* Stanford Machine Learning Class in Coursera 
* UFLDL Tutorial 
* Tensor Flow 
