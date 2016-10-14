#include <cassert>
#include <iomanip>
#include <iostream>

#include "../Common/MLUtils.h"
#include "SVMModel.h"

// Arguments:
//     n -- Number of input features.
//     c -- Coefficient of penalizing term
//     epsilon -- Slack values
//     ker -- Type of kernel to use (linear or gaussian only for now).
//     sigma -- Sigma value for gaussian filter.
SVMModel::SVMModel(
        unsigned int n,
        float c,
        float epsilon,
        SVMKernel ker,
        float sigma) :
    IModel<int>(n, true), c_(c), epsilon_(epsilon), sigma_(sigma), ker_(ker),
    b_(0), deltaB_(0)
{
}

SVMModel::~SVMModel()
{
}

// Inner implementation of training current model
void SVMModel::train_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y)
{
    clear();

    unsigned int m = x.rows();
    Eigen::MatrixXf xNorm = normalizerPtr_->normalizeTrainingData(x);

    alpha_ = Eigen::VectorXf::Zero(m);
    errorCache_ = Eigen::VectorXf::Zero(m);

    // precalculate dot products for gaussian kernel
    if (ker_ == SVMKernel::SVM_GAUSSIAN) {
        dotProdCache_.resize(m);
        for (unsigned int i = 0; i < m; ++i) {
            dotProdCache_[i] = (xNorm.row(i) * xNorm.row(i).transpose())[0];
        }

        // set landmarks as all normalized x values
        landmarksX_ = xNorm;
        landmarksY_ = y;
    }

    int numChanged = 0;
    int examineAll = 1;

    while (numChanged > 0 || examineAll) {
        numChanged = 0;
        if (examineAll) {
            for (unsigned int k = 0; k < m; ++k) {
                numChanged += examineExample_(xNorm, y, k);
            }
        } else {
            for (unsigned int k = 0; k < m; ++k) {
                if (alpha_[k] != 0 && alpha_[k] != c_) {
                    numChanged += examineExample_(xNorm, y, k);
                }
            }
        }

        if (examineAll == 1) {
            examineAll = 0;
        } else if (numChanged == 0) {
            examineAll = 1;
        }

        float s = 0, t = 0, obj = 0;

        for (unsigned int i = 0; i < m; ++i) {
            s += alpha_[i];
        }

        for (unsigned int i = 0; i < m; ++i) {
            for (unsigned int j = 0; j < m; ++j) {
                t += alpha_[i] * alpha_[j] * y[i] * y[j] * kernel_(xNorm, i, j);
            }
        }

        obj = s - 0.5 * t;
        std::cout << std::setprecision(5)
                  << "Objective func: " << obj << "\t\t\t"
                  << "Error rate: " << errorRate_(xNorm, y)
                  << std::endl;

        for (unsigned int i = 0; i < m; ++i) {
            if (alpha_[i] < 1e-6) {
                alpha_[i] = 0;
            }
        }
    }
}

// Inner implementation of predicting using current model
int SVMModel::predict_(const Eigen::VectorXf &x) const
{
    Eigen::VectorXf xNorm = normalizerPtr_->normalizeDataPoint(x);
    float s = 0;

    if (ker_ == SVMKernel::SVM_LINEAR) {
        s = dotProduct(w_, x);
    } else {
        for (unsigned int i = 0; i < landmarksX_.rows(); ++i) {
            float k = gaussianKernel(landmarksX_.row(i).transpose(), xNorm, sigma_,
                dotProdCache_[i]);
            s += alpha_[i] * landmarksY_[i] * k;
        }
    }
    s -= b_;

    return s >= 0 ? 1 : -1;
}

float SVMModel::kernel_(const Eigen::MatrixXf &x, int i1, int i2) const
{
    float k = 0;

    if (ker_ == SVMKernel::SVM_LINEAR) {
        k = linearKernel(x.row(i1).transpose(), x.row(i2).transpose());
    } else {
        k = gaussianKernel(x.row(i1).transpose(), x.row(i2).transpose(), sigma_,
            dotProdCache_[i1], dotProdCache_[i2]);
    }

    return k;
}

float SVMModel::errorRate_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y) const
{
    int nError = 0;
    unsigned int m = x.rows();

    for (unsigned int i = 0; i < m; ++i) {
        float learned = learnedFunc_(x, y, i);
        if ((learned >= 0 && y[i] < 0) || (learned < 0 && y[i] > 0)) {
            ++nError;
        }
    }
    return 1.0 * nError / m;
}

// Returns current learned value of k'th row of x
float SVMModel::learnedFunc_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y, int k) const
{
    assert(x.rows() == y.rows());
    float s = 0;

    if (ker_ == SVMKernel::SVM_LINEAR) {
        s = dotProduct(w_, x.row(k).transpose());
    } else {
        for (unsigned int i = 0; i < x.rows(); ++i) {
            if (alpha_[i] > 0) {
                float kVal = gaussianKernel(x.row(k).transpose(), x.row(i).transpose(),
                    sigma_, dotProdCache_[k], dotProdCache_[i]);
                s += alpha_[i] * y[i] * kVal;
            }
        }
    }
    s -= b_;

    return s;
}

// Checks whether alpha_i (i1) violates the KKT condition by more than TOLERANCE
//
// If it does, looks for second alpha_i (i2) and jointly optimizes both alpha_i's
// by calling takeStep_
//
// Code taken from: https://github.com/mazefeng/svm/blob/master/svm_solver.cpp (Baidu.com)
int SVMModel::examineExample_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y, int i1)
{
    float y1 = 0.0;
    float alpha1 = 0.0;
    float e1 = 0.0;
    float r1 = 0.0;
	int m = x.rows();

    y1 = y[i1];
    alpha1 = alpha_[i1];
    if (alpha1 > 0 && alpha1 < c_){
        e1 = errorCache_[i1];
    }
    else{
        e1 = learnedFunc_(x, y, i1) - y1;
    }

    r1 = y1 * e1;
    if ((r1 < -TOLERANCE && alpha1 < c_) || (r1 > TOLERANCE && alpha1 > 0)) {
        int k0 = 0;
        int k = 0;
        int i2 = -1;
        float tmax = 0.0;
        for (i2 = -1, tmax = 0, k = 0; k < m; k++) {
            if (alpha_[k] > 0 && alpha_[k] < c_){
                float e2 = 0.0;
                float temp = 0.0;
                e2 = errorCache_[k];
                temp = fabs(e1 - e2);
                if (temp > tmax) {
                    tmax = temp;
                    i2 = k;
                }
            }
            if (i2 >= 0) {
                if (takeStep_(x, y, i1, i2)){
                    return 1;
                }
            }
        }
        for (k0 = (int)(drand48() * m), k = k0; k < m + k0; k++) {
            i2 = k % m;
            if (alpha_[i2] > 0 && alpha_[i2] < c_) {
                if (takeStep_(x, y, i1, i2)){
                    return 1;
                }
            }
        }
        for (k0 = (int)(drand48() * m), k = k0; k < m + k0; k++){
            i2 = k % m;
            if (takeStep_(x, y, i1, i2)){
                return 1;
            }
        }
    }
    return 0;
}

// Optimizes two Lagrange multipliers
//
// Code taken from: https://github.com/mazefeng/svm/blob/master/svm_solver.cpp (Baidu.com)
int SVMModel::takeStep_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y, int i1, int i2)
{
    int y1 = 0;
    int y2 = 0;
    int s = 0;
    float alpha1 = 0.0;
    float alpha2 = 0.0;
    float a1 = 0.0;
    float a2 = 0.0;
    float e1 = 0.0;
    float e2 = 0.0;
    float low = 0.0;
    float high = 0.0;
    float k11 = 0.0;
    float k22 = 0.0;
    float k12 = 0.0;
    float eta = 0.0;
    float low_obj = 0.0;
    float high_obj = 0.0;
    if (i1 == i2){
        return 0;
    }

    alpha1 = alpha_[i1];
    alpha2 = alpha_[i2];
    y1 = y[i1];
    y2 = y[i2];

    if (alpha1 > 0 && alpha1 < c_){
        e1 = errorCache_[i1];
    }
    else{
        e1 = learnedFunc_(x, y, i1) - y1;
    }
    if (alpha2 > 0 && alpha2 < c_){
        e2 = errorCache_[i2];
    }
    else{
        e2 = learnedFunc_(x, y, i2) - y2;
    }
    s = y1 * y2;
    if (y1 == y2) {
        float gamma = alpha1 + alpha2;
        if (gamma > c_) {
            low = gamma - c_;
            high = c_;
        }
        else {
            low = 0;
            high = gamma;
        }
    }
    else{
        float gamma = alpha1 - alpha2;
        if (gamma > 0){
            low = 0;
            high = c_ - gamma;
        }
        else{
            low = -gamma;
            high = c_;
        }
    }

    if (fabs(low - high) < 1e-6){
        return 0;
    }

    k11 = kernel_(x, i1, i1);
    k12 = kernel_(x, i1, i2);
    k22 = kernel_(x, i2, i2);
    eta = 2 * k12 - k11 - k22;

    if (eta < 0) {
        a2 = alpha2 + y2 * (e2 - e1) / eta;
        if (a2 < low){
            a2 = low;
        }
        else if (a2 > high){
            a2 = high;
        }
    }
    else {
        float c1 = eta / 2.0;
        float c2 = y2 * (e1 - e2) - eta * alpha2;
        low_obj = c1 * low * low + c2 * low;
        high_obj = c1 * high * high + c2 * high;
        if (low_obj > high_obj + epsilon_){
            a2 = low;
        }
        else if (low_obj < high_obj - epsilon_){
            a2 = high;
        }
        else{
            a2 = alpha2;
        }
    }

    if (fabs(a2 - alpha2) < epsilon_ * (a2 + alpha2 + epsilon_)){
        return 0;
    }
    a1 = alpha1 - s * (a2 - alpha2);
    if (a1 < 0) {
        a2 += s * a1;
        a1 = 0;
    }
    else if (a1 > c_){
        float t = a1 - c_;
        a2 += s * t;
        a1 = c_;
    }

    float b1 = 0.0;
    float b2 = 0.0;
    float bnew = 0.0;
    if (a1 > 0 && a1 < c_){
        bnew = b_ + e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
    }
    else if (a2 > 0 && a2 < c_){
        bnew = b_ + e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
    }
    else{
        b1 = b_ + e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
        b2 = b_ + e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
        bnew = (b1 + b2) / 2.0;
    }

    deltaB_ = bnew - b_;
    b_ = bnew;

    float t1 = y1 * (a1 - alpha1);
    float t2 = y2 * (a2 - alpha2);

    if (ker_ == SVMKernel::SVM_LINEAR){
        w_ = w_ + t1 * x.row(i1).transpose() + t2 * x.row(i2).transpose();
    }

    for (int i = 0; i < x.rows(); i++){
        if (alpha_[i] > 0 && alpha_[i] < c_){
            errorCache_[i] += t1 * kernel_(x, i1, i) + t2 * kernel_(x, i2, i) - deltaB_;
        }
    }

    errorCache_[i1] = 0.0;
    errorCache_[i2] = 0.0;

    alpha_[i1] = a1;
    alpha_[i2] = a2;
    return 1;
}

// Clears all internal data for model (except user settings)
void SVMModel::clear()
{
    alpha_.resize(0);
    errorCache_.resize(0);
    dotProdCache_.resize(0);
    w_.resize(0);
    b_ = 0;
    deltaB_ = 0;

    landmarksX_.resize(0, 0);
    landmarksY_.resize(0);
}
