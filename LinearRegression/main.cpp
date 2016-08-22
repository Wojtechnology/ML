#include <iostream>

#include "LinearRegressionModel.h"

int main()
{
    std::vector<float> x;
    std::vector<float> y;

    int m;
    float newX, newY;
    std::cout << "m: ";
    std::cin >> m;

    for (float i = 0; i < m; ++i) {
        std::cin >> newX >> newY;
        x.push_back(newX);
        y.push_back(newY);
    }

    float alpha;
    int iterations;
    std::cout << "alpha: ";
    std::cin >> alpha;
    std::cout << "iterations: ";
    std::cin >> iterations;

    LinearRegressionModel<float> model;
    model.train(x, y, alpha, iterations);

    float query;
    while (std::cin >> query) {
        std::cout << model.predict(query) << std::endl;
    }
}
