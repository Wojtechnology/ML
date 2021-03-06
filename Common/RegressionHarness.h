#ifndef REGRESSION_HARNESS_H
#define REGRESSION_HARNESS_H

#include <iostream>

#include <Eigen/Dense>

#define TRAIN_PRINT_FREQUENCY 100000000
#define TEST_PRINT_FREQUENCY 100000000

// O - Output type
// M - Model type
template <typename O, typename M>
void regressionHarness(int argc, char **argv)
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
    Eigen::Matrix<O, Eigen::Dynamic, 1> y(m);

    float placeholder;
    O oPlaceholder;

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

    M model(n, normalize, alpha, iterations, lambda);
    model.train(x, y);

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
        float errorMargin;
        O res, prediction;

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

#endif
