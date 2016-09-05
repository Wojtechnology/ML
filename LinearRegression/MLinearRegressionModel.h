#ifndef M_LINEAR_REGRESSION_MODEL_H
#define M_LINEAR_REGRESSION_MODEL_H

#include <Eigen/Dense>

#include "../Common/IRegressionModel.h"

// Multivariate Linear Regression model for float type values
class MLinearRegressionModel : public IRegressionModel<float> {
public:
    explicit MLinearRegressionModel(unsigned int n, bool normalize = false) :
             IRegressionModel(n, normalize) { }

    void train(const Eigen::MatrixXf &x,
               const Eigen::VectorXf &y,
               float alpha = 1,
               unsigned int iterations = 100);
    float predict(const Eigen::VectorXf &x) const;
};

#endif
