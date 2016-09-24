#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "NeuralNetworkModel.h"

int main(int argc, char **argv)
{
    int numLayers = 4;
    std::vector<int> layerSizes = {2, 100, 100, 1};
    NeuralNetworkModel model(numLayers, layerSizes);

    Eigen::MatrixXf x(5, 2);
    Eigen::MatrixXf y(5, 1);
    x << 1, 1,
         2, 2,
         3, 3,
         4, 4,
         5, 5;
    y << 1,
         2,
         3,
         4,
         5;

    model.train(x, y, 0.01, 1000, 0.01);

}
