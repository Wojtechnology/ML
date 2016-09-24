#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "NeuralNetworkModel.h"

int main(int argc, char **argv)
{
    int numLayers = 3;
    std::vector<int> layerSizes = {2, 2, 1};
    NeuralNetworkModel model(numLayers, layerSizes);

    Eigen::MatrixXf x(4, 2);
    Eigen::MatrixXf y(4, 1);
    x << 0, 0,
         0, 1,
         1, 0,
         1, 1,
    y << 1,
         0,
         0,
         1,

    model.train(x, y, 1, 1000, 0);
    model.print();

    float first, second;
    while (std::cin >> first) {
        std::cin >> second;
        Eigen::VectorXf x(2);
        x << first, second;
        std::cout << model.predict(x) << std::endl;
    }
}
