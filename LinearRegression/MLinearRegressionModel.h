#ifndef M_LINEAR_REGRESSION_MODEL_H
#define M_LINEAR_REGRESSION_MODEL_H

#include <Eigen/Dense>

#include "../Common/IRegressionModel.h"

// Multivariate Linear Regression model for float type values
class MLinearRegressionModel : public IRegressionModel<float> {
public:
    explicit MLinearRegressionModel(
            unsigned int n,
            bool normalize = true,
            float alpha = 1,
            unsigned int iterations = 1,
            float lambda = 1) :
        IRegressionModel(n, normalize, alpha, iterations, lambda) { }

private:
    void train_(const Eigen::MatrixXf &x, const Eigen::VectorXf &y) override;
    float predict_(const Eigen::VectorXf &x) const override;
};

#endif
