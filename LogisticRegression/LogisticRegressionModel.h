#ifndef LOGISTIC_REGRESSION_MODEL_H
#define LOGISTIC_REGRESSION_MODEL_H

#include <string>

#include <Eigen/Dense>

// Binary logistic regressing model
class LogisticRegressionModel {
public:
    LogisticRegressionModel(unsigned int n, bool normalize = false);
    void train(const Eigen::MatrixXf &x,
               const Eigen::MatrixXb &y,
               float alpha = 1,
               unsigned int iterations = 100);
    void train(const Eigen::VectorXf &x) const;
    void dump(const std::string &path);
    void load(const std::string &path);
private:
    Eigen::VectorXf theta_;
    int n_;
}

#endif
