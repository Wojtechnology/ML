#ifndef LINEAR_REGRESSION_MODEL_H
#define LINEAR_REGRESSION_MODEL_H

#include <vector>

// Model for training and predicting using simple linear regression
template <typename T = float>
class LinearRegressionModel {
public:
    explicit LinearRegressionModel(float theta0 = 0, float theta1 = 0);
    void train(const std::vector<T> &x,
               const std::vector<T> &y,
               float alpha = 1,
               unsigned int iterations = 100);
    T predict(T x) const;

private:
    float theta0_;
    float theta1_;
};

#endif
