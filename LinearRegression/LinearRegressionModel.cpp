#include <cassert>
#include <iostream>

#include "LinearRegressionModel.h"

// Constructor to set initial values
// Defaults to theta1 = 0, theta0 = 0
template <typename T>
LinearRegressionModel<T>::LinearRegressionModel(float theta0, float theta1) :
    theta0_(theta0), theta1_(theta1)
{ }

// Function to train the parameters of the model using gradient descent
template <typename T>
void LinearRegressionModel<T>::train(const std::vector<T> &x,
                                     const std::vector<T> &y,
                                     float alpha,
                                     unsigned int iterations)
{
    assert(x.size() == y.size());
    unsigned int m = x.size();
    float sum0, sum1, delta;
    for (unsigned int i = 0; i < iterations; ++i) {
        sum0 = 0.0;
        sum1 = 0.0;
        // calculate sum of partial derivatives
        for (unsigned int j = 0; j < m; ++j) {
             delta = predict(x.at(j)) - y.at(j);
             sum0 += delta;
             sum1 += delta * x.at(j);
        }
        // update theta
        theta0_ = theta0_ - alpha*(sum0/m);
        theta1_ = theta1_ - alpha*(sum1/m);
        if ((i+1) % 100 == 0) {
            std::cout << "Iteration " << (i+1) << ": y = ";
            std::cout << theta1_ << " * x + " << theta0_ << std::endl;
        }
    }
}

// Function to predict value using current model
template <typename T>
T LinearRegressionModel<T>::predict(T x) const
{
    return (T) (theta0_ + theta1_ * x);
}

template class LinearRegressionModel<int>;
template class LinearRegressionModel<float>;
