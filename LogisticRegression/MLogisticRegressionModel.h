#ifndef M_LOGISTIC_REGRESSION_MODEL_H
#define M_LOGISTIC_REGRESSION_MODEL_H

#include <vector>

#include <Eigen/Dense>

#include "../Common/IModel.h"
#include "LogisticRegressionModel.h"

// Multiclass logistic regression model
// 0 is first class, 1 is second, etc
class MLogisticRegressionModel : public IModel<int> {
public:
    MLogisticRegressionModel(
        unsigned int n,
        unsigned int numClasses,
        bool normalize = true,
        float alpha = 1,
        unsigned int iterations = 100,
        float lambda = 1);
    ~MLogisticRegressionModel();

private:
    void train_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y) override;
    int predict_(const Eigen::VectorXf &x) const override;

    static Eigen::VectorXi isolateClass_(const Eigen::VectorXi &x, int cl);

    unsigned int numClasses_;
    std::vector<LogisticRegressionModel*> models_;

};

#endif
