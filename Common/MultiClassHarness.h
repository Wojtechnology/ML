#ifndef MULTI_CLASS_HARNESS_H
#define MULTI_CLASS_HARNESS_H

#include <iostream>

#include <Eigen/Dense>

#include "../LogisticRegression/MLogisticRegressionModel.h"

#define TRAIN_PRINT_FREQUENCY 100000000

void multiClassHarness(int argc, char **argv)
{
    int m, n, numClasses;

    bool normalize = false;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == 'n') normalize = true;
    }

    std::cout << "m: ";
    std::cin >> m;
    std::cout << "n: ";
    std::cin >> n;
    std::cout << "numClasses: ";
    std::cin >> numClasses;

    Eigen::MatrixXf x(m,n);
    Eigen::VectorXi y(m);

    float placeholder;
    int oPlaceholder;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> placeholder;
            x(i, j) = placeholder;
        }
        std::cin >> oPlaceholder;
        y(i) = oPlaceholder;
        if ((i+1) % TRAIN_PRINT_FREQUENCY == 0) {
            std::cout << "Loaded " << (i+1) << "th train point" << std::endl;
        }
    }

    float alpha, lambda = 0;
    unsigned int iterations;

    std::cout << "alpha: ";
    std::cin >> alpha;
    std::cout << "iterations: ";
    std::cin >> iterations;
    std::cout << "lambda: ";
    std::cin >> lambda;

    MLogisticRegressionModel model(n, numClasses, normalize, alpha, iterations, lambda);
    model.train(x, y);

    Eigen::VectorXf query(n);
    while (true) {
        for (int i = 0; i < n; ++i) {
            std::cin >> placeholder;
            query(i) = placeholder;
        }
        if (std::cin.fail()) break;
        std::cout << model.predict(query) << std::endl;
    }
}

#endif
