#include "../Common/MLUtils.h"
#include "SVMKernel.h"

float linearKernel(const Eigen::VectorXf &x1, const Eigen::VectorXf &x2)
{
    return dotProduct(x1, x2);
}

float gaussianKernel(
    const Eigen::VectorXf &x1, const Eigen::VectorXf &x2, float sigma,
    float x1DotX1, float x2DotX2)
{
    float x1DotX2 = dotProduct(x1, x2);
    if (isnan(x1DotX1)) {
        x1DotX1 = dotProduct(x1, x1);
    }
    if (isnan(x2DotX2)) {
        x2DotX2 = dotProduct(x2, x2);
    }

    float k = x1DotX1 + x2DotX2 - 2 * x1DotX2;
    k /= -(2.0 * sigma * sigma);
    k = exp(k);

    return k;
}
