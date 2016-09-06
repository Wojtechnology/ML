#include <iostream>
#include <math.h>

#include "LogisticRegressionModel.h"

#define ITERATION_PRINT_FREQUENCY 100

// implementation for training model using gradient descent
// uses sigmoid function
void LogisticRegressionModel::train_(const Eigen::MatrixXf &x,
                                     const Eigen::VectorXi &y,
                                     float alpha,
                                     unsigned int iterations)
{
    unsigned int m = x.rows();

    Eigen::MatrixXf trainingData(m, n_+1);
    Eigen::VectorXf ones = Eigen::VectorXf::Constant(m, 1); // vector of 1's
    trainingData << ones, (normalizerPtr_ ? normalizerPtr_->normalizeTrainingData(x) : x);

    Eigen::VectorXf output(m);
    for (unsigned int i = 0; i < m; ++i) {
        output[i] = y[i] ? 1.0 : 0.0;
    }

    for (unsigned int i = 0; i < iterations; ++i) {
        theta_ -= alpha*((sigmoid_(trainingData * theta_) - output).transpose() * trainingData).transpose() / m;
        if ((i+1) % ITERATION_PRINT_FREQUENCY == 0) {
            std::cout << "Iteration " << (i+1) << ": ϴ = [";
            std::cout << theta_.transpose() << "]\n";
        }
    }
}

// implementation for predicting outcome using current model
int LogisticRegressionModel::predict_(const Eigen::VectorXf &x) const
{
    Eigen::RowVectorXf query(x.rows()+1);
    // append x_0
    query << 1, (normalizerPtr_ ? normalizerPtr_->normalizeDataPoint(x) : x).transpose();
    return sigmoid_((query * theta_)[0]) > 0.5 ? 1 : 0;
}

// sigmoid function applied to a vector
Eigen::VectorXf LogisticRegressionModel::sigmoid_(const Eigen::VectorXf &v)
{
    return v.unaryExpr<float(*)(float)>(&LogisticRegressionModel::sigmoid_);
}

// sigmoid function applied to a float
float LogisticRegressionModel::sigmoid_(float x)
{
    return 1.0 / (1.0 + exp(-x));
}
