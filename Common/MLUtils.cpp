#include <math.h>

#include "MLUtils.h"

namespace MLUtils {

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

};
