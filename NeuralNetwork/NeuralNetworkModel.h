#ifndef NEURAL_NETWORK_MODEL_H
#define NEURAL_NETWORK_MODEL_H

#include <vector>

#include <Eigen/Dense>

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
    // [ y1
    //   y2
    //  ....
    //   ym ]
    //
    void train(const Eigen::MatrixXf &x,
               const Eigen::VectorXf &y,
               float alpha,
               unsigned int iterations,
               float lambda);

    // predict using current model
    //
    // x in the form:
    // [ x_1
    //   x_2
    //   ...
    //   x_n ]
    //
    Eigen::VectorXf predict(const Eigen::VectorXf &x);

private:
    std::vector<Eigen::MatrixXf> thetas_;
    int inputSize_;
    int outputSize_;
    int numLayers_;

    // helper methods
    void initializeThetas_();
    std::vector<Eigen::VectorXf> forwardProp_(const Eigen::VectorXf &x);
    std::deque<Eigen::VectorXf> backProp_(
        const Eigen::VectorXf &y, const std::vector<Eigen::VectorXf> &a);
};

#endif
