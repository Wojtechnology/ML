#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include <Eigen/Dense>

// SVM model that implements the SMO (Sequential Minimal Optimization) algorithm
// A lot of ideas are borrowed from https://github.com/mazefeng/svm
//
// Originally, this method was proposed by the following paper:
//     Platt, J., "Sequential Minimal Optimization:
//     A Fast Algorithm for Training Support Vector Machines." (1998).
class SVMModel {
public:
    SVMModel();
    ~SVMModel();

    // train the model using given training data
    //
    // x in the form:
    // [ --- x1 ---
    //   --- x2 ---
    //      ....
    //   --- xm --- ]
    //
    // y in the form:
    // [ y1
    //   y2
    //  ....
    //   ym ]
    //
    void train();

    // predict using current model
    //
    // x in the form:
    // [ x_1
    //   x_2
    //   ...
    //   x_n ]
    //
    float predict();

private:


};

#endif
