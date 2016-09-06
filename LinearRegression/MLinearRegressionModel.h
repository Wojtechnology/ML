#ifndef M_LINEAR_REGRESSION_MODEL_H
#define M_LINEAR_REGRESSION_MODEL_H

#include <Eigen/Dense>

#include "../Common/IRegressionModel.h"

// Multivariate Linear Regression model for float type values
class MLinearRegressionModel : public IRegressionModel<float> {
public:
    explicit MLinearRegressionModel(unsigned int n, bool normalize = false) :
             IRegressionModel(n, normalize) { }

private:
    void train_(const Eigen::MatrixXf &x,
                const Eigen::VectorXf &y,
                float alpha,
                unsigned int iterations) override;
    float predict_(const Eigen::VectorXf &x) const override;
};

#endif
