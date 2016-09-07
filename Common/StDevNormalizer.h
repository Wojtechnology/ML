#ifndef ST_DEV_NORMALIZER_H
#define ST_DEV_NORMALIZER_H

#include "../Common/INormalizer.h"

class StDevNormalizer : public INormalizer {
public:
    explicit StDevNormalizer(unsigned int n);
    Eigen::MatrixXf normalizeTrainingData(const Eigen::MatrixXf &x) override;
    Eigen::VectorXf normalizeDataPoint(const Eigen::VectorXf &x) const override;
};

#endif
