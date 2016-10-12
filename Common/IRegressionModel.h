#ifndef I_REGRESSION_MODEL_H
#define I_REGRESSION_MODEL_H

#include <Eigen/Dense>

#include "IModel.h"

// Interface for regression models
// T is output type
template <typename T>
class IRegressionModel : public IModel<T> {
public:
    virtual ~IRegressionModel() { }
protected:
    explicit IRegressionModel(
            unsigned int n,
            bool normalize = true,
            float alpha = 1,
            unsigned iterations = 100,
            float lambda = 1) :
        IModel<T>(n, normalize), theta_(Eigen::VectorXf::Zero(n+1)), alpha_(alpha),
        iterations_(iterations), lambda_(lambda)
    {
    }

    // devalues all weights in theta_ except theta_0
    void regularizeTheta_(float alpha, float lambda, unsigned int m)
    {
        for (unsigned int i = 1; i < theta_.rows(); ++i) {
            theta_[i] = (1 - alpha * lambda / m) * theta_[i];
        }
    }

    Eigen::VectorXf theta_;
    float alpha_;
    unsigned int iterations_;
    float lambda_;
};

#endif
