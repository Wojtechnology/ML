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
            bool normalize = false,
            bool regularize = false) :
        IRegressionModel(n, normalize, regularize) { }

    // returns the probability for the positive class
    float predictProb(const Eigen::VectorXf &x) const;

private:
    void train_(const Eigen::MatrixXf &x,
                const Eigen::VectorXi &y,
                float alpha,
                unsigned int iterations,
                float lambda) override;
    int predict_(const Eigen::VectorXf &x) const override;
};

#endif
