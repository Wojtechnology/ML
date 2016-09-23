#ifndef NEURAL_NETWORK_MODEL_H
#define NEURAL_NETWORK_MODEL_H

#include <vector>

#include <Eigen/Dense>

class NeuralNetworkModel {
public:
    NeuralNetworkModel(int numLayers, const std::vector<int> &layerSizes);

    Eigen::VectorXf predict(const Eigen::VectorXf &input);
private:
    std::vector<Eigen::MatrixXf> thetas_;
    int inputSize_;
    int outputSize_;
    int numLayers_;

    // helper methods
    void initializeThetas_();
    std::vector<Eigen::VectorXf> forwardProp_(const Eigen::VectorXf &input);
};

#endif
