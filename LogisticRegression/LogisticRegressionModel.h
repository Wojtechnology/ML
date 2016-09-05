#ifndef LOGISTIC_REGRESSION_MODEL_H
#define LOGISTIC_REGRESSION_MODEL_H

#include <string>

#include <Eigen/Dense>

#include "../Common/IRegressionModel.h"

// Binary logistic regressing model
class LogisticRegressionModel : public IRegressionModel<int> {
public:
    explicit LogisticRegressionModel(unsigned int n, bool normalize = false) :
             IRegressionModel(n, normalize) { }

    void train(const Eigen::MatrixXf &x,
               const Eigen::MatrixXi &y,
               float alpha = 1,
               unsigned int iterations = 100);
    int predict(const Eigen::VectorXf &x) const;
};

#endif
