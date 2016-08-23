#include <cassert>
#include <iostream>

#include "MultivariateLinearRegressionModel.h"

// initialize theta
MultivariateLinearRegressionModel::MultivariateLinearRegressionModel(unsigned int n,
                                                                     bool normalize) :
    theta_(Eigen::VectorXf::Zero(n+1)), n_(n), normalize_(normalize)
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
    trainingData << ones, x;

    for (unsigned int i = 0; i < iterations; ++i) {
        theta_ -= alpha*((trainingData * theta_ - y).transpose() * trainingData).transpose() / m;
        if ((i+1) % 100 == 0) {
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
    query << 1, x.transpose();
    return query * theta_;
}
