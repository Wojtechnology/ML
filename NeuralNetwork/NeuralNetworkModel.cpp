#include <cassert>
#include <deque>
#include <math.h>

#include "../Common/MLUtils.h"
#include "NeuralNetworkModel.h"

NeuralNetworkModel::NeuralNetworkModel(int numLayers, const std::vector<int> &layerSizes) :
    thetas_(numLayers-1), numLayers_(numLayers)
{
    // Need at least input and output layers
    assert(numLayers >= 2);
    assert(numLayers == layerSizes.size());
    inputSize_ = layerSizes[0];
    outputSize_ = layerSizes[numLayers-1];

    for (int i = 0; i < numLayers-1; ++i) {
        thetas_[i].resize(layerSizes[i+1], layerSizes[i]+1);
    }
    initializeThetas_();
}

NeuralNetworkModel::~NeuralNetworkModel()
{
}

void train(const Eigen::MatrixXf &x,
           const Eigen::VectorXf &y,
           float alpha,
           int iterations,
           float lambda)
{
    for (int i = 0; i < iterations; ++i) {
        // 1) forward propagation to calculate activation values and output layer
        // 2) back propagation to calculate error terms and sum to total error
        // 3) calculate partial derivatives
        // 4) gradient descent
    }
}

// Returns output layer after running forward propagation
Eigen::VectorXf NeuralNetworkModel::predict(const Eigen::VectorXf &x)
{
    assert(x.rows() == inputSize_);
    std::vector<Eigen::VectorXf> a = forwardProp_(x);
    return a[numLayers_-1];
}

// Randomly initializes thetas (using Hugo Larochelle, Glorot & Bengio (2010) method)
// Uses uniform distribution with bounds +/- sqrt(6/(inNodes + outNodes))
void NeuralNetworkModel::initializeThetas_()
{
    // Initialize seed
    srand (time(NULL));

    for (int i = 0; i < numLayers_-1; ++i) {
        int outNodes = thetas_[i].rows();
        // NB: including bias unit as input
        int inNodes = thetas_[i].cols();
        float bound = sqrt(6 / (((float) inNodes) + ((float) outNodes)));

        for (int j = 0; j < outNodes; ++j) {
            for (int k = 0; k < inNodes; ++k) {
                float randInit = - bound + ((float) rand()) / ((float) (RAND_MAX/(2 * bound)));
                thetas_[i](j, k) = randInit;
            }
        }
    }
}

// Returns all layers after running forward propagation
std::vector<Eigen::VectorXf> NeuralNetworkModel::forwardProp_(const Eigen::VectorXf &x)
{
    assert(x.rows() == inputSize_);
    std::vector<Eigen::VectorXf> a(1);
    a[0] = x;
    for (int i = 0; i < numLayers_-1; ++i) {
        // Add bias unit
        Eigen::VectorXf curLayer(a[i].rows()+1);
        curLayer << 1, a[i];
        a.push_back(MLUtils::sigmoid(thetas_[i] * curLayer));
    }

    return a;
}

// Returns all deltas for a particular training example
std::deque<Eigen::VectorXf> NeuralNetworkModel::backProp_(
    const Eigen::VectorXf &y, const std::vector<Eigen::VectorXf> &a)
{
    assert(y.rows() == outputSize_);
    std::deque<Eigen::VectorXf> d(1);

    return d;
}
