#include <iostream>

#include "LinearRegressionModel.h"

int main()
{
    std::vector<float> x;
    std::vector<float> y;

    int m;
    float newX, newY;
    std::cin >> m;

    for (int i = 0; i < m; ++i) {
        std::cin >> newX >> newY;
        x.push_back(newX);
        y.push_back(newY);
    }

    LinearRegressionModel<float> model;
    model.train(x, y, 0.01);

    float query;
    while (std::cin >> query) {
        std::cout << model.predict(query) << std::endl;
    }
}
