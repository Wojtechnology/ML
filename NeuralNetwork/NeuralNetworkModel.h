#ifndef NEURAL_NETWORK_MODEL_H
#define NEURAL_NETWORK_MODEL_H

#include <deque>
#include <vector>

#include <Eigen/Dense>

#include "../Common/INormalizer.h"

class NeuralNetworkModel {
public:
    NeuralNetworkModel(int numLayers, const std::vector<int> &layerSizes);
    ~NeuralNetworkModel();

    // train the model using given training data
    //
    // x in the form:
    // [ --- x1 ---
    //   --- x2 ---
    //      ....
    //   --- xm --- ]
    //
    // y in the form:
    // [ --- y1 ---
    //   --- y2 ---
    //      ....
    //   --- ym --- ]
    //
    // 1) forward propagation to calculate activation values
    // 2) back propagation to calculate error terms and sum to total error
    // 3) calculate partial derivatives
    // 4) gradient descent
    // 5) repeat until hopefully finding some minima
    void train(const Eigen::MatrixXf &x,
               const Eigen::MatrixXf &y,
               float alpha,
               int iterations,
               float lambda = 0);

    // predict using current model
    //
    // x in the form:
    // [ x_1
    //   x_2
    //   ...
    //   x_n ]
    //
    Eigen::VectorXf predict(const Eigen::VectorXf &x);

    // prints the current thetas between all layers
    void print();

private:
    std::vector<Eigen::MatrixXf> thetas_;
    int inputSize_;
    int outputSize_;
    int numLayers_;
    INormalizer *normalizerPtr_;

    // helper methods
    void initializeThetas_();
    std::vector<Eigen::MatrixXf> thetasWithoutBias_();
    std::vector<Eigen::MatrixXf> deltaZeros_();
    std::vector<Eigen::VectorXf> forwardProp_(const Eigen::VectorXf &x);
    std::deque<Eigen::VectorXf> backProp_(
        const Eigen::VectorXf &y, const std::vector<Eigen::VectorXf> &a);

};

#endif
