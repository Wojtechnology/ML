#include <cmath>

#include "StDevNormalizer.h"

StDevNormalizer::StDevNormalizer(unsigned int n) : INormalizer(n)
{
}

// normalizes training set and sets the mean and standard deviation vectors
Eigen::MatrixXf StDevNormalizer::normalizeTrainingData(
        const Eigen::MatrixXf &x)
{
    assert(x.cols() == n_);
    unsigned int m = x.rows();
    for (int i = 0; i < n_; ++i) {
        means_(i) = x.col(i).mean();
        range_(i) = std::sqrt((x.col(i).array() - means_(i)).pow(2).sum()/m);
    }
    return (x.transpose() - means_.replicate(1, x.rows())).cwiseQuotient(range_.replicate(1, x.rows())).transpose();
}

// normalizes a data point with previous mean and range
Eigen::VectorXf StDevNormalizer::normalizeDataPoint(
        const Eigen::VectorXf &x) const
{
    assert(x.rows() == n_);
    return (x - means_).cwiseQuotient(range_);
}
