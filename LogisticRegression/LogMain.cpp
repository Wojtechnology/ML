#include "../Common/RegressionHarness.h"
#include "LogisticRegressionModel.h"

int main(int argc, char **argv)
{
    regressionHarness<int, LogisticRegressionModel>(argc, argv);
}
