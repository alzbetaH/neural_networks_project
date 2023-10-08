#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>

using namespace std;
class Neuron;

typedef vector<Neuron> Layer;

struct Connection
{
    float weight;
    float deltaweight;
};

class Neuron
{
private:
    static float eta; //[0.0...1.0] overrall learning rate
    static float alpha; //[0.0...n] multiplier of last weight change (momentum) 
    static float transferFunction(float x);
    static float transferFunctionDerivative(float x);
    static float heWeightInit(unsigned numNetworkInputs);
    float sumDOW(const Layer &nextLayer) const;
    float m_outVal;
    vector<Connection> m_outWeights;
    unsigned m_myIndex;
    float m_gradient;

public:
    Neuron(unsigned numNetworkInputs, unsigned numNeuronOutputs, unsigned neuronIndex);
    void setOutputVal(float val) { m_outVal = val;}
    float getOutputVal(void) const { return m_outVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(float targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

    static void setLearningRate(float learningRate);
};