#include <cassert>
#include <iostream>

#include "MLinearRegressionModel.h"

#define ITERATION_PRINT_FREQUENCY 100

// train the model using given training data
//
// x in the form:
// [ --- x1 ---
//   --- x2 ---
//      ....
//   --- xm --- ]
//
// y in the form:
// [ y1
//   y2
//  ....
//   ym ]
//
void MLinearRegressionModel::train(const Eigen::MatrixXf &x,
                                              const Eigen::VectorXf &y,
                                              float alpha,
                                              unsigned int iterations)
{
    // assert that number of data points is equal for x and y
    assert(x.rows() == y.rows());

    // assert that number of features provided is allowed
    assert(x.cols() == n_);

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

// predict using current model
//
// x in the form:
// [ x_1
//   x_2
//   ...
//   x_n ]
//
float MLinearRegressionModel::predict(const Eigen::VectorXf &x) const
{
    assert(x.rows() == n_);
    Eigen::RowVectorXf query(x.rows()+1);
    // append x_0
    query << 1, (normalizerPtr_ ? normalizerPtr_->normalizeDataPoint(x) : x).transpose();
    return query * theta_;
}
