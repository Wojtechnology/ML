#ifndef I_REGRESSION_MODEL_H
#define I_REGRESSION_MODEL_H

#include <string>

#include <Eigen/Dense>

#include "../Common/INormalizer.h"

// Interface for regression models
class IRegressionModel {
public:
    virtual void train(const Eigen::MatrixXf &x,
                       const Eigen::VectorXf &y,
                       float alpha = 1,
                       unsigned int iterations = 100) = 0;
    virtual float predict(const Eigen::VectorXf &x) const = 0;

    // The following two functions do not have to be implemented
    virtual void dump(const std::string &path) const
    {
        throw "not implemented";
    }

    virtual void load(const std::string &path)
    {
        throw "not implemented";
    }

    virtual ~IRegressionModel();

protected:
    explicit IRegressionModel(unsigned int n, bool normalize = false);

    Eigen::VectorXf theta_;
    unsigned int n_; // number of features

    INormalizer *normalizerPtr_;
};

#endif
