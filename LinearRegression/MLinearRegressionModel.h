#ifndef M_LINEAR_REGRESSION_MODEL_H
#define M_LINEAR_REGRESSION_MODEL_H

#include <Eigen/Dense>

#include "../Common/IRegressionModel.h"

// Multivariate Linear Regression model for float type values
class MLinearRegressionModel : public IRegressionModel<float> {
public:
    explicit MLinearRegressionModel(
            unsigned int n,
            bool normalize = false,
            bool regularize = false) :
        IRegressionModel(n, normalize, regularize) { }

private:
    void train_(const Eigen::MatrixXf &x,
                const Eigen::VectorXf &y,
                float alpha,
                unsigned int iterations,
                float lambda) override;
    float predict_(const Eigen::VectorXf &x) const override;
};

#endif
