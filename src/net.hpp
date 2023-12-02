#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "neuron.hpp"

using namespace std;

class Net
{
private:
    vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;

public:
    Net(const vector<unsigned> &topology, unsigned seed);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void updateWeights();
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const {return m_recentAverageError;};
    double getError(void) const {return m_error;};
    void calcAvgGradient(unsigned int batchSize);
    void resetGradientSum();
    double getLoss(const vector<double> &targetVals);
    double validationAccuracy();
    int compare_result(const vector<double> &output, const vector<double> &label);
    void setDropout(unsigned int layer_num, double probability);
};
