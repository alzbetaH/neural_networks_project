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
    static float heWeightInit(unsigned numLayerInputs);
    float sumDOW(const Layer &nextLayer) const;
    float m_potential; // Inner neuron potential
    float m_outVal; // Output value of the neuron after activation function
    vector<float> m_outWeights; // A weight float for each neuron in the next layer // TODO make private
    vector<float> m_outWeightsDeltas; // Previous weight change for each weight - for momentum
    vector<float> m_outWeightsGradients;
    unsigned m_myIndex; // Index of this neuron in the layer
    float m_gradient;

public:
    Neuron(unsigned numLayerInputs, unsigned numNeuronOutputs, unsigned neuronIndex);
    float getPotential() { return m_potential; }
    void setOutputVal(float val) { m_outVal = val; }
    float getOutputVal(void) const { return m_outVal; }
    void calcPotential(const Layer &prevLayer);
    void calcOutput();
    void calcOutputGradients(float targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void calcWeightGradients(const Layer &nextLayer);
    void updateWeights();
    void calcAvgGradient(unsigned int batchSize);
    void resetGradientSum();

    static void setLearningRate(float learningRate);
};