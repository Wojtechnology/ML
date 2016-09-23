#ifndef ML_UTILS_H
#define ML_UTILS_H

#include <Eigen/Dense>

namespace MLUtils {

    Eigen::VectorXf sigmoid(const Eigen::VectorXf &v);
    float sigmoid(float x);

};

#endif
