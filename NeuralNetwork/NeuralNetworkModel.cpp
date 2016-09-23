#include <cassert>

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

// Returns output layer after running forward propagation
Eigen::VectorXf NeuralNetworkModel::predict(const Eigen::VectorXf &input)
{
    assert(input.rows() == inputSize_);
    std::vector<Eigen::VectorXf> layers = forwardProp_(input);
    return layers[numLayers_-1];
}

// Randomly initializes thetas
void NeuralNetworkModel::initializeThetas_()
{
    thetas_[0] << -30, 20, 20,
                  10, -20, -20;
    thetas_[1] << -10, 20, 20;
}

// Returns all layers after running forward propagation
std::vector<Eigen::VectorXf> NeuralNetworkModel::forwardProp_(const Eigen::VectorXf &input)
{
    assert(input.rows() == inputSize_);
    std::vector<Eigen::VectorXf> layers(1);
    layers[0] = input;
    for (int i = 0; i < numLayers_-1; ++i) {
        // Add bias unit
        Eigen::VectorXf curLayer(layers[i].rows()+1);
        curLayer << 1, layers[i];
        layers.push_back(MLUtils::sigmoid(thetas_[i] * curLayer));
    }

    return layers;
}
