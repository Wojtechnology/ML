#ifndef MULTIVARIATE_LINEAR_REGRESSION_MODEL_H
#define MULTIVARIATE_LINEAR_REGRESSION_MODEL_H

#include <Eigen/Dense>

// Multivariate Linear Regression model for float type values
class MultivariateLinearRegressionModel {
public:
    explicit MultivariateLinearRegressionModel(unsigned int n, bool normalize = false);
    void train(const Eigen::MatrixXf &x,
               const Eigen::VectorXf &y,
               float alpha = 1,
               unsigned int iterations = 100);
    float predict(const Eigen::VectorXf &x) const;

private:
    Eigen::VectorXf theta_;
    int n_; // number of features

    bool normalize_;
};

#endif
