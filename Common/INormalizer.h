#ifndef I_NORMALIZER_H
#define I_NORMALIZER_H

#include <Eigen/Dense>

class INormalizer {
public:
    explicit INormalizer(unsigned int n) : n_(n), means_(n), range_(n) { }
    virtual ~INormalizer() { }

    virtual Eigen::MatrixXf normalizeTrainingData(const Eigen::MatrixXf &x) = 0;
    virtual Eigen::VectorXf normalizeDataPoint(const Eigen::VectorXf &x) const = 0;

protected:
    unsigned int n_; // number of features

    Eigen::VectorXf means_; // means for normalization
    Eigen::VectorXf range_; // ranges for normalization
};

#endif
