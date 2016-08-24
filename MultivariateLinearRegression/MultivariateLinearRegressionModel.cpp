#include <cassert>
#include <iostream>

#include "MultivariateLinearRegressionModel.h"

#define ITERATION_PRINT_FREQUENCY 100

// initialize theta
MultivariateLinearRegressionModel::MultivariateLinearRegressionModel(unsigned int n,
                                                                     bool normalize) :
    theta_(Eigen::VectorXf::Zero(n+1)), n_(n), normalize_(normalize), means_(n), range_(n)
{
}

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
void MultivariateLinearRegressionModel::train(const Eigen::MatrixXf &x,
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
    trainingData << ones, (normalize_ ? normalizeTrainingData(x) : x);

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
float MultivariateLinearRegressionModel::predict(const Eigen::VectorXf &x) const
{
    assert(x.rows() == n_);
    Eigen::RowVectorXf query(x.rows()+1);
    // append x_0
    query << 1, (normalize_ ? normalizeDataPoint(x) : x).transpose();
    return query * theta_;
}

// normalizes training set and sets the mean and standard deviation vectors
Eigen::MatrixXf MultivariateLinearRegressionModel::normalizeTrainingData(
        const Eigen::MatrixXf &x)
{
    assert(x.cols() == n_);
    for (int i = 0; i < n_; ++i) {
        means_(i) = x.col(i).mean();
        // TODO: Change to using stdev
        range_(i) = x.col(i).maxCoeff() - x.col(i).minCoeff();
    }
    return (x.transpose() - means_.replicate(1, x.rows())).cwiseQuotient(range_.replicate(1, x.rows())).transpose();
}

// normalizes a data point with previous mean and range
Eigen::VectorXf MultivariateLinearRegressionModel::normalizeDataPoint(
        const Eigen::VectorXf &x) const
{
    assert(x.rows() == n_);
    return (x - means_).cwiseQuotient(range_);
}
