#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include "neuron.hpp"

using namespace std;

class Net
{
private:
    vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    float m_error;
    float m_recentAverageError;
    static float m_recentAverageSmoothingFactor;

public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<float> &inputVals);
    void backProp(const vector<float> &targetVals);
    void updateWeights();
    void getResults(vector<float> &resultVals) const;
    float getRecentAverageError(void) const {return m_recentAverageError;};
    float getError(void) const {return m_error;};
    void calcAvgGradient(unsigned int batchSize);
    void resetGradientSum();
};
