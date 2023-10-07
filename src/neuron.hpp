#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>

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
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outVal;
    vector<Connection> m_outWeights;
    unsigned m_myIndex;
    double m_gradient;

public:
    Neuron(unsigned numOutVals, unsigned myIndex);
    void setOutputVal(double val) { m_outVal = val;}
    double getOutputVal(void) const { return m_outVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
};