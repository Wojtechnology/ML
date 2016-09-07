#ifndef I_REGRESSION_MODEL_H
#define I_REGRESSION_MODEL_H

#include <cassert>
#include <string>

#include <Eigen/Dense>

#include "INormalizer.h"
#include "StDevNormalizer.h"

// Interface for regression models
// T is output type
template <typename T>
class IRegressionModel {
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;

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
    void train(const Eigen::MatrixXf &x,
               const VectorXT &y,
               float alpha = 1,
               unsigned int iterations = 100)
    {
        // assert that number of data points is equal for x and y
        assert(x.rows() == y.rows());

        // assert that number of features provided is allowed
        assert(x.cols() == n_);

        // use template implementation
        train_(x, y, alpha, iterations);
    }

    // predict using current model
    //
    // x in the form:
    // [ x_1
    //   x_2
    //   ...
    //   x_n ]
    //
    T predict(const Eigen::VectorXf &x) const
    {
        // assert that the number of features provided is allowed
        assert(x.rows() == n_);

        // use template implementation
        return predict_(x);
    }

    virtual ~IRegressionModel()
    {
        if (normalizerPtr_) {
            delete normalizerPtr_;
        }
    }

protected:
    explicit IRegressionModel(unsigned int n, bool normalize = false) :
        theta_(Eigen::VectorXf::Zero(n+1)), n_(n), normalizerPtr_(nullptr)
    {
        if (normalize) {
            normalizerPtr_ = new StDevNormalizer(n);
        }
    }

    virtual void train_(const Eigen::MatrixXf &x,
                        const VectorXT &y,
                        float alpha,
                        unsigned int iterations) = 0;

    virtual T predict_(const Eigen::VectorXf &x) const = 0;

    Eigen::VectorXf theta_;
    unsigned int n_; // number of features

    INormalizer *normalizerPtr_;
};

#endif
