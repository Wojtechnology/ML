#ifndef I_REGRESSION_MODEL_H
#define I_REGRESSION_MODEL_H

#include <string>

#include <Eigen/Dense>

#include "INormalizer.h"
#include "RangeNormalizer.h"

// Interface for regression models
// T is output type
template <typename T>
class IRegressionModel {
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXT;

    virtual void train(const Eigen::MatrixXf &x,
                       const VectorXT &y,
                       float alpha = 1,
                       unsigned int iterations = 100) = 0;
    virtual T predict(const Eigen::VectorXf &x) const = 0;

    // The following two functions do not have to be implemented
    virtual void dump(const std::string &path) const
    {
        throw "not implemented";
    }

    virtual void load(const std::string &path)
    {
        throw "not implemented";
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
            normalizerPtr_ = new RangeNormalizer(n);
        }
    }

    Eigen::VectorXf theta_;
    unsigned int n_; // number of features

    INormalizer *normalizerPtr_;
};

#endif
