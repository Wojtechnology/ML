#include <iostream>

#include "../Common/MLUtils.h"
#include "LogisticRegressionModel.h"

#define ITERATION_PRINT_FREQUENCY 100

// implementation for training model using gradient descent
// uses sigmoid function
void LogisticRegressionModel::train_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y)
{
    unsigned int m = x.rows();

    Eigen::MatrixXf trainingData(m, n_+1);
    Eigen::VectorXf ones = Eigen::VectorXf::Constant(m, 1); // vector of 1's
    trainingData << ones, (normalizerPtr_ ? normalizerPtr_->normalizeTrainingData(x) : x);

    Eigen::VectorXf output(m);
    for (unsigned int i = 0; i < m; ++i) {
        output[i] = y[i] ? 1.0 : 0.0;
    }

    for (unsigned int i = 0; i < iterations_; ++i) {
        // calculate change to each parameter (using partial derivatives)
        Eigen::VectorXf delta = alpha_*((sigmoid(trainingData * theta_) - output).transpose() * trainingData).transpose() / m;
        regularizeTheta_(alpha_, lambda_, m);
        theta_ -= delta;

        if ((i+1) % ITERATION_PRINT_FREQUENCY == 0) {
            std::cout << "Iteration " << (i+1) << ": ϴ = [";
            std::cout << theta_.transpose() << "]\n";
        }
    }
}

float LogisticRegressionModel::predictProb(const Eigen::VectorXf &x) const
{
    Eigen::RowVectorXf query(x.rows()+1);
    // append x_0
    query << 1, (normalizerPtr_ ? normalizerPtr_->normalizeDataPoint(x) : x).transpose();
    return sigmoid((query * theta_)[0]);
}

// implementation for predicting outcome using current model
int LogisticRegressionModel::predict_(const Eigen::VectorXf &x) const
{
    return predictProb(x) >= 0.5 ? 1 : 0;
}
