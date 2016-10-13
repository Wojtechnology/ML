#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include <Eigen/Dense>

#include "../Common/IModel.h"
#include "SVMKernel.h"

#define TOLERANCE 1e-6

// SVM model that implements the SMO (Sequential Minimal Optimization) algorithm
// The code at https://github.com/mazefeng/svm was modified to work with Eigen
//
// Originally, this method was proposed by the following paper:
//     Platt, J., "Sequential Minimal Optimization:
//     A Fast Algorithm for Training Support Vector Machines." (1998).
class SVMModel : public IModel<int> {
public:
    SVMModel(
        unsigned int n,
        float c,
        float epsilon,
        SVMKernel kernel = SVMKernel::SVM_LINEAR,
        float sigma = 1);
    ~SVMModel();

    void clear();

private:
    void train_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y) override;
    int predict_(const Eigen::VectorXf &x) const override;

    float kernel_(const Eigen::MatrixXf &x, int i1, int i2) const;
    float learnedFunc(const Eigen::MatrixXf &x, int k) const;

    // SMO implementation
    int examineExample_(const Eigen::MatrixXf &x, int i1);
    int takeStep_(const Eigen::MatrixXf &x, int i1, int i2);


    // User options
    float c_;
    float epsilon_;
    float sigma_;
    SVMKernel ker_;

    // Internal options
    Eigen::VectorXf alpha_;
    Eigen::VectorXf error_cache_;
    Eigen::VectorXf dot_prod_cache_;
    Eigen::VectorXf w_;
    float b_;
    float deltaB_;

    // used as landmarks for gaussian kernel
    Eigen::MatrixXf landmarksX_;
    Eigen::VectorXi landmarksY_;
};

#endif
