#include "RangeNormalizer.h"

RangeNormalizer::RangeNormalizer(unsigned int n) : INormalizer(n)
{
}

// normalizes training set and sets the mean and standard deviation vectors
Eigen::MatrixXf RangeNormalizer::normalizeTrainingData(
        const Eigen::MatrixXf &x)
{
    assert(x.cols() == n_);
    for (int i = 0; i < n_; ++i) {
        means_(i) = x.col(i).mean();
        // TODO: Change to using stdev
        range_(i) = x.col(i).maxCoeff() - x.col(i).minCoeff();
    }
    return (x.transpose() - means_.replicate(1, x.rows())).cwiseQuotient(range_.replicate(1, x.rows())).transpose();
}

// normalizes a data point with previous mean and range
Eigen::VectorXf RangeNormalizer::normalizeDataPoint(
        const Eigen::VectorXf &x) const
{
    assert(x.rows() == n_);
    return (x - means_).cwiseQuotient(range_);
}
