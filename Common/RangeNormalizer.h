#ifndef RANGE_NORMALIZER_H
#define RANGE_NORMALIZER_H

#include "../Common/INormalizer.h"

class RangeNormalizer : public INormalizer {
public:
    explicit RangeNormalizer(unsigned int n);
    Eigen::MatrixXf normalizeTrainingData(const Eigen::MatrixXf &x) override;
    Eigen::VectorXf normalizeDataPoint(const Eigen::VectorXf &x) const override;
};

#endif
