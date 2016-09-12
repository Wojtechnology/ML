#ifndef M_LOGISTIC_REGRESSION_MODEL_H
#define M_LOGISTIC_REGRESSION_MODEL_H

#include <vector>

#include <Eigen/Dense>

#include "LogisticRegressionModel.h"

// Multiclass logistic regression model
// 0 is first class, 1 is second, etc
class MLogisticRegressionModel {
public:
    MLogisticRegressionModel(
            unsigned int n,
            unsigned int numClasses,
            bool normalize = false,
            bool regularize = false);
    ~MLogisticRegressionModel();

    void train(const Eigen::MatrixXf &x,
               const Eigen::VectorXi &y,
               float alpha = 1,
               unsigned int iterations = 100,
               float lambda = 1);
    int predict(const Eigen::VectorXf &x) const;

private:
    static Eigen::VectorXi isolateClass_(const Eigen::VectorXi &x, int cl);

    unsigned int numClasses_;
    std::vector<LogisticRegressionModel*> models_;

};

#endif
