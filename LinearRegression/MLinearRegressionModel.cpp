#include <iostream>

#include "MLinearRegressionModel.h"

#define ITERATION_PRINT_FREQUENCY 100

// implementation for training model using gradient descent
// cost function is squared difference
void MLinearRegressionModel::train_(const Eigen::MatrixXf &x,
                                    const Eigen::VectorXf &y,
                                    float alpha,
                                    unsigned int iterations)
{
    unsigned int m = x.rows();

    Eigen::MatrixXf trainingData(m, n_+1);
    Eigen::VectorXf ones = Eigen::VectorXf::Constant(m, 1); // vector of 1's
    trainingData << ones, (normalizerPtr_ ? normalizerPtr_->normalizeTrainingData(x) : x);

    for (unsigned int i = 0; i < iterations; ++i) {
        theta_ -= alpha*((trainingData * theta_ - y).transpose() * trainingData).transpose() / m;
        if ((i+1) % ITERATION_PRINT_FREQUENCY == 0) {
            std::cout << "Iteration " << (i+1) << ": ϴ = [";
            std::cout << theta_.transpose() << "]\n";
        }
    }
}

// implementation for predicting outcome using current model
float MLinearRegressionModel::predict_(const Eigen::VectorXf &x) const
{
    Eigen::RowVectorXf query(x.rows()+1);
    // append x_0
    query << 1, (normalizerPtr_ ? normalizerPtr_->normalizeDataPoint(x) : x).transpose();
    return query * theta_;
}
