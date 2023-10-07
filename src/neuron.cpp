#include "neuron.hpp"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer &prevLayer)
{
    //including bias
    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        Neuron &neuron = prevLayer[i];
        double oldDeltaWeight = neuron.m_outWeights[m_myIndex].deltaweight;

        double newDeltaWeight =
            //eta = learning rate
            eta 
            * neuron.getOutputVal()
            * m_gradient
            //momentum
            + alpha
            * oldDeltaWeight;
        neuron.m_outWeights[m_myIndex].deltaweight = newDeltaWeight;
        neuron.m_outWeights[m_myIndex].weight += newDeltaWeight;
    }
    
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    for (unsigned i = 0; i < nextLayer.size() - 1; ++i)
    {
        sum += m_outWeights[i].weight * nextLayer[i].m_gradient;
    }
    
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outVal);
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    // 1.0 - x * x is derivative of tanh(x)
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        sum += prevLayer[i].getOutputVal() * prevLayer[i].m_outWeights[m_myIndex].weight;
    }
    m_outVal = Neuron::transferFunction(sum);
}


Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c)
    {
        m_outWeights.push_back(Connection());
        m_outWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
    
}