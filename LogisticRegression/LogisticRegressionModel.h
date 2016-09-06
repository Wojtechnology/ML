#ifndef LOGISTIC_REGRESSION_MODEL_H
#define LOGISTIC_REGRESSION_MODEL_H

#include <string>

#include <Eigen/Dense>

#include "../Common/IRegressionModel.h"

// Binary logistic regressing model
// Output: if 0: negative class
//         else: positive class
class LogisticRegressionModel : public IRegressionModel<int> {
public:
    explicit LogisticRegressionModel(unsigned int n, bool normalize = false) :
             IRegressionModel(n, normalize) { }

private:
    void train_(const Eigen::MatrixXf &x,
                const Eigen::VectorXi &y,
                float alpha = 1,
                unsigned int iterations = 100) override;
    int predict_(const Eigen::VectorXf &x) const override;

    static Eigen::VectorXf sigmoid_(const Eigen::VectorXf &v);
    static float sigmoid_(float x);
};

#endif
