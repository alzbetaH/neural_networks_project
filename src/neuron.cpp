#include "neuron.hpp"

float Neuron::eta = 0.15;
float Neuron::alpha = 0.1;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c)
    {
        m_outWeights.push_back(Connection());
        m_outWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    //including bias
    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        Neuron &neuron = prevLayer[i];
        float oldDeltaWeight = neuron.m_outWeights[m_myIndex].deltaweight;

        // - (Learning rate * prev neuron output * gradient) + momentum * old weight change
        float newDeltaWeight =
            -(eta * neuron.getOutputVal() * m_gradient) + alpha * oldDeltaWeight;

        neuron.m_outWeights[m_myIndex].deltaweight = newDeltaWeight;
        neuron.m_outWeights[m_myIndex].weight += newDeltaWeight;
    }
}

float Neuron::sumDOW(const Layer &nextLayer) const
{
    /**
     * @brief Sum of derivatives of weights
     */
    float sum = 0.0;

    for (unsigned i = 0; i < nextLayer.size() - 1; ++i)
    {
        sum += m_outWeights[i].weight * nextLayer[i].m_gradient;
    }
    
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    float dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outVal);
}

void Neuron::calcOutputGradients(float targetVal)
{
    float delta = m_outVal - targetVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outVal);
}

float Neuron::transferFunction(float x)
{
    return tanh(x);

    // return x < 0 ? 0 : x; // ReLU !Doesn't work
}

float Neuron::transferFunctionDerivative(float x)
{
    return 1.0 - x * x; // approximate derivative of tanh

    // return x < 0 ? 0 : 1; // ReLU !Doesn't work
}

void Neuron::feedForward(const Layer &prevLayer)
{
    float sum = 0.0;

    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        sum += prevLayer[i].getOutputVal() * prevLayer[i].m_outWeights[m_myIndex].weight;
    }
    m_outVal = Neuron::transferFunction(sum);
}

void Neuron::setLearningRate(float learningRate)
{
    eta = learningRate;
}