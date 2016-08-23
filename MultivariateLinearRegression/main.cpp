#include <iostream>

#include <Eigen/Dense>

#include "MultivariateLinearRegressionModel.h"

int main()
{
    int m, n;

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
    }

    MultivariateLinearRegressionModel model(n);

    float alpha;
    unsigned int iterations;

    std::cout << "alpha: ";
    std::cin >> alpha;
    std::cout << "iterations: ";
    std::cin >> iterations;

    model.train(x, y, alpha, iterations);

    Eigen::VectorXf query(n);
    while (true) {
        for (int i = 0; i < n; i++) {
            std::cin >> placeholder;
            query(i) = placeholder;
        }
        if (std::cin.fail()) break;
        std::cout << model.predict(query) << std::endl;
    }
}
