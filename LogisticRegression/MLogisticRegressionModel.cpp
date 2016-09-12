#include <cassert>

#include "MLogisticRegressionModel.h"

MLogisticRegressionModel::MLogisticRegressionModel(
        unsigned int n,
        unsigned int numClasses,
        bool normalize,
        bool regularize) :
    numClasses_(numClasses), models_(numClasses)
{
    assert(numClasses >= 2);
    for (unsigned int i = 0; i < numClasses; ++i) {
        models_[i] = new LogisticRegressionModel(n, normalize, regularize);
    }
}

MLogisticRegressionModel::~MLogisticRegressionModel()
{
    for (unsigned int i = 0; i < numClasses_; ++i) {
        delete models_[i];
    }
}

void MLogisticRegressionModel::train(const Eigen::MatrixXf &x,
                                     const Eigen::VectorXi &y,
                                     float alpha,
                                     unsigned int iterations,
                                     float lambda)
{
    for (unsigned int i = 0; i < numClasses_; ++i) {
        models_[i]->train(x, isolateClass_(y, i), alpha, iterations, lambda);
    }
}

int MLogisticRegressionModel::predict(const Eigen::VectorXf &x) const
{
    unsigned int bestClass = 0;
    float highestProb = 0;
    for (unsigned int i = 0; i < numClasses_; ++i) {
        float prob = models_[i]->predict(x);
        if (prob > highestProb) {
            highestProb = prob;
            bestClass = i;
        }
    }
    return bestClass;
}

Eigen::VectorXi MLogisticRegressionModel::isolateClass_(const Eigen::VectorXi &y, int cl)
{
    unsigned int m = y.rows();
    Eigen::VectorXi result(m);
    for (unsigned int i = 0; i < m; ++i) {
        result[i] = (y[i] == cl) ? 1 : 0;
    }
    return result;
}
