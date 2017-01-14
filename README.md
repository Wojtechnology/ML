# ML
Naive machine learning algorithm implementations in c++. I built these while taking
Andrew Ng's machine learning course on Coursera.

## Models
- Linear Regression
- Logistic Regression (Binary and Multiclass)
- Neural Network
- SVM (https://github.com/mazefeng/svm/blob/master/svm_solver.cpp)

## How to run
To run the code examples you will need to have [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) installed.

### Linear Regression
Navigate to the `LinearRegression` directory and run:
```
make
./mlinreg n < data/mdata1.txt
```
This will train a multivariate linear regression model on some data. The `n` flag tells the program to normalize the input data.

Other examples are found in the `data` subdirectory.

### Logistic Regression
Navigate to the `LogisticRegression` directory and run:
```
make
./logreg n < data/data2.txt
```
This will train a logistic regression model on some data. The `n` flag tells the program to normalize the input data.

Other examples are found in the `data` subdirectory.

### Neural Network
Navigate to the `NeuralNetwork` directory and run:
```
make
./nn d data/digits.txt
```
This will train a neural network to predict a digit from a 20x20 input image.

### Support Vector Machine
Navigate to the `SVM` directory and run:
```
make
./svm < data/data2.txt
```
This will train a SVM model on some data.

Other examples are found in the `data` subdirectory.
