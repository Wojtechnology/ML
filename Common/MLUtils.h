#ifndef ML_UTILS_H
#define ML_UTILS_H

#include <Eigen/Dense>

Eigen::VectorXf sigmoid(const Eigen::VectorXf &v);
float sigmoid(float x);

float dotProduct(const Eigen::VectorXf &x1, const Eigen::VectorXf &x2);

#endif
