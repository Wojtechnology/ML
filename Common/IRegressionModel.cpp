#include <cassert>

#include "../Common/RangeNormalizer.h"
#include "IRegressionModel.h"

// create normalizer object if requested
IRegressionModel::IRegressionModel(unsigned int n, bool normalize):
    theta_(Eigen::VectorXf::Zero(n+1)), n_(n), normalizerPtr_(nullptr)
{
    if (normalize) {
        normalizerPtr_ = new RangeNormalizer(n);
    }
}

// delete normalizer pointer if it exists
IRegressionModel::~IRegressionModel() {
    if (normalizerPtr_) {
        delete normalizerPtr_;
    }
}
