#include "../Common/HarnessUtils.h"
#include "MLinearRegressionModel.h"

int main(int argc, char **argv)
{
    regressionHarness<float, MLinearRegressionModel>(argc, argv);
}
