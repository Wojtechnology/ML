#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Dense>

#include "MultivariateLinearRegressionModel.h"

#define TRAIN_PRINT_FREQUENCY 100000000
#define TEST_PRINT_FREQUENCY 100000000

int main(int argc, char **argv)
{
    int m, n;

    bool normalize = false, test = false;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == 'n') normalize = true;
        if (argv[i][0] == 't') test = true;
    }

    std::cout << "m: ";
    std::cin >> m;
    std::cout << "n: ";
    std::cin >> n;

    Eigen::MatrixXf x(m,n);
    Eigen::VectorXf y(m);

    float placeholder;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> placeholder;
            x(i, j) = placeholder;
        }
        std::cin >> placeholder;
        y(i) = placeholder;
        if ((i+1) % TRAIN_PRINT_FREQUENCY == 0) {
            std::cout << "Loaded " << (i+1) << "th train point" << std::endl;
        }
    }

    MultivariateLinearRegressionModel model(n, normalize);

    float alpha;
    unsigned int iterations;

    std::cout << "alpha: ";
    std::cin >> alpha;
    std::cout << "iterations: ";
    std::cin >> iterations;

    model.train(x, y, alpha, iterations);

    Eigen::VectorXf query(n);
    if (!test) {
        while (true) {
            for (int i = 0; i < n; ++i) {
                std::cin >> placeholder;
                query(i) = placeholder;
            }
            if (std::cin.fail()) break;
            std::cout << model.predict(query) << std::endl;
        }
    } else {
        int numTests, correct = 0;
        float errorMargin, res, prediction;

        std::cout << "num tests: ";
        std::cin >> numTests;
        std::cout << "error margin: ";
        std::cin >> errorMargin;

        for (int i = 0; i < numTests; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cin >> placeholder;
                query(j) = placeholder;
            }
            std::cin >> res;
            if ((i+1) % TRAIN_PRINT_FREQUENCY == 0) {
                std::cout << "Loaded " << (i+1) << "th test point" << std::endl;
            }
            prediction = model.predict(query);
            if (std::abs(res - prediction) <= errorMargin) ++correct;
        }

        std::cout << correct << " out of " << numTests << " correct" << std::endl;
    }
}
