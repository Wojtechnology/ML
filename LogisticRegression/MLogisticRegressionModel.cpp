#include <cassert>

#include "MLogisticRegressionModel.h"

MLogisticRegressionModel::MLogisticRegressionModel(
        unsigned int n,
        unsigned int numClasses,
        bool normalize,
        float alpha,
        unsigned int iterations,
        float lambda) :
    IModel<int>(n, normalize), numClasses_(numClasses), models_(numClasses)
{
    assert(numClasses >= 2);
    for (unsigned int i = 0; i < numClasses; ++i) {
        // Set normalize to false since we already normalize here
        models_[i] = new LogisticRegressionModel(n, false, alpha, iterations, lambda);
    }
}

MLogisticRegressionModel::~MLogisticRegressionModel()
{
    for (unsigned int i = 0; i < numClasses_; ++i) {
        delete models_[i];
    }
}

void MLogisticRegressionModel::train_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y)
{
    Eigen::MatrixXf trainingData = normalizerPtr_ ? normalizerPtr_->normalizeTrainingData(x) : x;
    for (unsigned int i = 0; i < numClasses_; ++i) {
        models_[i]->train(trainingData, isolateClass_(y, i));
    }
}

int MLogisticRegressionModel::predict_(const Eigen::VectorXf &x) const
{
    Eigen::VectorXf query = normalizerPtr_ ? normalizerPtr_->normalizeDataPoint(x) : x;
    unsigned int bestClass = 0;
    float highestProb = 0;
    for (unsigned int i = 0; i < numClasses_; ++i) {
        float prob = models_[i]->predictProb(query);
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
