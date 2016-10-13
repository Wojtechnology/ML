#ifndef SVM_KERNEL_H
#define SVM_KERNEL_H

#include <math.h>

#include <Eigen/Dense>

enum SVMKernel {
    SVM_LINEAR,
    SVM_GAUSSIAN
};

// Calculates linear kernel
float linearKernel(const Eigen::VectorXf &x1, const Eigen::VectorXf &x2);

// Calculates gaussian kernel
//     ||x1 − x2||^2 = x1Tx1 + x2Tx2 - 2x1Tx2
//
// Optionally, provide dot products
float gaussianKernel(
    const Eigen::VectorXf &x1, const Eigen::VectorXf &x2, float sigma,
    float x1DotX1 = NAN, float x2DotX2 = NAN);

#endif
