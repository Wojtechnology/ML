#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "NeuralNetworkModel.h"

int main(int argc, char **argv)
{
    int numLayers = 3;
    std::vector<int> layerSizes = {2, 2, 1};
    NeuralNetworkModel model(numLayers, layerSizes);

    float first, second;
    while (std::cin >> first) {
        std::cin >> second;
        Eigen::VectorXf input(2);
        input << first, second;
        std::cout << model.predict(input) << std::endl;
    }
}
