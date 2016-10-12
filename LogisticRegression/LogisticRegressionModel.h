#ifndef LOGISTIC_REGRESSION_MODEL_H
#define LOGISTIC_REGRESSION_MODEL_H

#include <string>

#include <Eigen/Dense>

#include "../Common/IRegressionModel.h"

// Binary logistic regression model
// Output: if 0: negative class
//         else: positive class
class LogisticRegressionModel : public IRegressionModel<int> {
public:
    explicit LogisticRegressionModel(
            unsigned int n,
            bool normalize = true,
            float alpha = 1,
            unsigned int iterations = 100,
            float lambda = 1) :
        IRegressionModel(n, normalize, alpha, iterations, lambda) { }

    // returns the probability for the positive class
    float predictProb(const Eigen::VectorXf &x) const;

private:
    void train_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y) override;
    int predict_(const Eigen::VectorXf &x) const override;
};

#endif
