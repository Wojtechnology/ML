#include <cassert>
#include <iostream>
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

void NeuralNetworkModel::train(const Eigen::MatrixXf &x,
                               const Eigen::MatrixXf &y,
                               float alpha,
                               int iterations,
                               float lambda)
{
    assert(x.cols() == inputSize_);
    assert(y.cols() == outputSize_);
    assert(x.rows() == y.rows());

    int m = x.rows();

    for (int i = 0; i < iterations; ++i) {
        std::vector<Eigen::MatrixXf> deltaSums = deltaZeros_();
        for (int j = 0; j < m; ++j) {
            // 1) forward propagation to calculate activation values
            std::vector<Eigen::VectorXf> a = forwardProp_(x.row(j).transpose());
            // 2) back propagation to calculate error terms and sum to total error
            std::deque<Eigen::VectorXf> d = backProp_(y.row(j).transpose(), a);
        }
    }
}

// Returns output layer after running forward propagation
Eigen::VectorXf NeuralNetworkModel::predict(const Eigen::VectorXf &x)
{
    assert(x.rows() == inputSize_);
    std::vector<Eigen::VectorXf> a = forwardProp_(x);
    return a[numLayers_-1];
}

void NeuralNetworkModel::print()
{
    std::cout << std::endl;
    for (int i = 0; i < numLayers_-1; ++i) {
        std::cout << "Theta from layer " << (i + 1) << " to ";
        std::cout << (i + 2) << ":\n";
        std::cout << thetas_[i];
        std::cout << std::endl << std::endl;
    }
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

// Returns zero matrices the same sizes as thetas_
std::vector<Eigen::MatrixXf> NeuralNetworkModel::deltaZeros_()
{
    std::vector<Eigen::MatrixXf> deltaSums;
    for (Eigen::MatrixXf theta : thetas_) {
        deltaSums.push_back(Eigen::MatrixXf::Zero(theta.rows(), theta.cols()));
    }
    return deltaSums;
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
        a[i] = curLayer;

        // Calculate activations
        a.push_back(MLUtils::sigmoid(thetas_[i] * a[i]));
    }

    return a;
}

// Returns all deltas for a particular training example
std::deque<Eigen::VectorXf> NeuralNetworkModel::backProp_(
    const Eigen::VectorXf &y, const std::vector<Eigen::VectorXf> &a)
{
    assert(y.rows() == outputSize_);
    assert(a.size() == numLayers_);

    std::deque<Eigen::VectorXf> d(1);
    d[0] = a[numLayers_-1] - y;
    for (int i = numLayers_-2; i > 0; --i) {
        // If not output layer, remove bias unit
        Eigen::VectorXf curD(d[0].rows() - (i == numLayers_-2 ? 0 : 1));
        if (i == numLayers_-2) {
            curD << d[0];
        } else {
            curD << d[0].block(1, 0, d[0].rows()-1, 1);
        }

        Eigen::VectorXf ones = Eigen::VectorXf::Ones(a[i].rows());
        d.push_front((thetas_[i].transpose() * curD).cwiseProduct(a[i].cwiseProduct(ones-a[i])));
    }
    return d;
}
