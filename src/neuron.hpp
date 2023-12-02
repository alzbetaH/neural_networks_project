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
    double weight;
    double deltaweight;
};

class Neuron
{
private:
    static double eta; //[0.0...1.0] overrall learning rate
    static double alpha; //[0.0...n] multiplier of last weight change (momentum) 
    static double decay;
    static double epsilon;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double heWeightInit(unsigned numLayerInputs);
    double sumDOW(const Layer &nextLayer) const;
    double m_potential; // Inner neuron potential
    double m_outVal; // Output value of the neuron after activation function
    vector<double> m_outWeights; // A weight double for each neuron in the next layer // TODO make private
    vector<double> m_outWeightsDeltas; // Previous weight change for each weight - for momentum
    vector<double> m_outWeightsGradients;
    unsigned m_myIndex; // Index of this neuron in the layer
    double m_gradient;
    int dropout_probability;
    int is_training;

public:
    Neuron(unsigned numLayerInputs, unsigned numNeuronOutputs, unsigned neuronIndex);
    void setDropout(double probability);
    double getPotential() { return m_potential; }
    void setOutputVal(double val) { m_outVal = val; }
    double getOutputVal(void) const { return m_outVal; }
    void calcPotential(const Layer &prevLayer);
    void calcOutput();
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void calcWeightGradients(const Layer &nextLayer);
    void updateWeights();
    void calcAvgGradient(unsigned int batchSize);
    void resetGradientSum();

    static void setLearningRate(double learningRate);
};