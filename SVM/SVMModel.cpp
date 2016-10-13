#include "../Common/MLUtils.h"
#include "SVMModel.h"

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

void SVMModel::train_(const Eigen::MatrixXf &x, const Eigen::VectorXi &y)
{
    unsigned int m = x.rows();
    Eigen::MatrixXf xNorm = normalizerPtr_->normalizeTrainingData(x);

    // precalculate dot products for gaussian kernel
    if (ker_ == SVMKernel::SVM_GAUSSIAN) {
        dot_prod_cache_.resize(m);
        for (unsigned int i = 0; i < m; ++i) {
            dot_prod_cache_[i] = (xNorm.row(i) * xNorm.row(i).transpose())[0];
        }

        // set landmarks as all normalized x values
        landmarksX_ = xNorm;
        landmarksY_ = y;
    }
}

int SVMModel::predict_(const Eigen::VectorXf &x) const
{
    Eigen::VectorXf xNorm = normalizerPtr_->normalizeDataPoint(x);

    float s = 0;
    if (ker_ == SVMKernel::SVM_LINEAR) {
        s = dotProduct(w_, x);
    } else {
        for (unsigned int i = 0; i < landmarksX_.rows(); ++i) {
            float k = gaussianKernel(landmarksX_.row(i).transpose(), x, sigma_,
                dot_prod_cache_[i]);
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
    }
    else {
        k = gaussianKernel(x.row(i1).transpose(), x.row(i2).transpose(), sigma_,
            dot_prod_cache_[i1], dot_prod_cache_[i2]);
    }

    return k;
}

int SVMModel::examineExample_(const Eigen::MatrixXf &x, int i1)
{
    return 0;
}

int SVMModel::takeStep_(const Eigen::MatrixXf &x, int i1, int i2)
{
    return 0;
}

// Clears all internal data for model (except user settings)
void SVMModel::clear()
{
    alpha_.resize(0, 0);
    error_cache_.resize(0, 0);
    dot_prod_cache_.resize(0, 0);
    w_.resize(0, 0);
    b_ = 0;
    deltaB_ = 0;

    landmarksX_.resize(0, 0);
    landmarksY_.resize(0);
}
