#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "NeuralNetworkModel.h"

void courseraGates()
{
    int numLayers = 3;
    std::vector<int> layerSizes = {2, 2, 1};
    NeuralNetworkModel model(numLayers, layerSizes);

    Eigen::MatrixXf X(4, 2);
    Eigen::MatrixXf y(4, 1);
    X << 0, 0,
         0, 1,
         1, 0,
         1, 1,
    y << 1,
         0,
         0,
         1,

    model.train(X, y, 1, 1000);
    model.print();

    float first, second;
    while (std::cin >> first) {
        std::cin >> second;
        Eigen::VectorXf x(2);
        x << first, second;
        std::cout << model.predict(x) << std::endl;
    }
}

int highestIndex(const Eigen::RowVectorXf &x)
{
    // best choice should be greater than 0
    assert(x.cols() > 0);
    float highest = x[0];
    int highestIdx = 0;
    for (int i = 1; i < x.cols(); ++i) {
        if (x[i] > highest) {
            highest = x[i];
            highestIdx = i;
        }
    }
    return highestIdx;
}

// TODO: write abstractions for data manip
void courseraDigits(char *path)
{
    // init model
    const int inputSize = 400;
    const int outputSize = 10;
    const int numLayers = 3;
    std::vector<int> layerSizes = {inputSize, 25, outputSize};
    NeuralNetworkModel model(numLayers, layerSizes);

    // load and shuffle data
    std::ifstream dataFile;
    dataFile.open(path);
    std::vector<Eigen::RowVectorXf> data;

    const int rowSize = inputSize + outputSize;
    const int numRows = 5000;
    float val;
    for (int i = 0; i < numRows; ++i) {
        Eigen::RowVectorXf row(rowSize);
        for (int j = 0; j < rowSize; ++j) {
            dataFile >> val;
            row[j] = val;
        }
        data.push_back(row);
    }
    std::random_shuffle(data.begin(), data.end());

    // separate training data from test data
    const int numTrain = 4500;
    assert(numTrain <= numRows);

    Eigen::MatrixXf XTrain(numTrain, inputSize);
    Eigen::MatrixXf yTrain(numTrain, outputSize);
    Eigen::MatrixXf XTest(numRows-numTrain, inputSize);
    Eigen::MatrixXf yTest(numRows-numTrain, outputSize);

    for (int i = 0; i < numRows; ++i) {
        if (i < numTrain) {
            XTrain.row(i) = data[i].block(0, 0, 1, inputSize);
            yTrain.row(i) = data[i].block(0, inputSize, 1, outputSize);
        } else {
            XTest.row(i-numTrain) = data[i].block(0, 0, 1, inputSize);
            yTest.row(i-numTrain) = data[i].block(0, inputSize, 1, outputSize);
        }
    }

    model.train(XTrain, yTrain, 10, 20, 0.01);

    int numCorrect = 0;
    for (int i = 0; i < numRows - numTrain; ++i) {
        std::cout << "Test: " << i + 1 << std::endl;
        int expected = highestIndex(yTest.row(i));
        Eigen::VectorXf result = model.predict(XTest.row(i).transpose()).transpose();
        int predicted = highestIndex(result);
        if (expected != predicted) {
            std::cout << "WRONG: " << expected << " != " << predicted << std::endl;
        } else {
            std::cout << "RIGHT: " << expected << " == " << predicted << std::endl;
            ++numCorrect;
        }
    }
    std::cout << "Report: " << numCorrect << "/" << numRows - numTrain << " correct\n";
    std::cout << "That's: " << float(numCorrect) / float(numRows-numTrain) * 100 << "%\n";
}

void printUsage()
{
    std::cout << "Usage: \n\tg for gates\n\t";
    std::cout << "d for digits (also provide data file as second arg)\n";
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cout << "No command provided\n";
        printUsage();
        exit(0);
    }

    if (argv[1][0] == 'g') {
        courseraGates();
    } else if (argv[1][0] == 'd') {
        if (argc < 3) {
            std::cout << "Didn't provide data file\n";
            printUsage();
        } else {
            courseraDigits(argv[2]);
        }
    } else {
        std::cout << "Unsupported command\n";
        printUsage();
    }
}
