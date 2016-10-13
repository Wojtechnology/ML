#include <math.h>

#include "MLUtils.h"

// sigmoid function applied to a vector
Eigen::VectorXf sigmoid(const Eigen::VectorXf &v)
{
    return v.unaryExpr<float(*)(float)>(&sigmoid);
}

// sigmoid function applied to a float
float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

// helper - dot product of two float vectors
float dotProduct(const Eigen::VectorXf &x1, const Eigen::VectorXf &x2)
{
    return (x1.transpose() * x2)[0];
}
