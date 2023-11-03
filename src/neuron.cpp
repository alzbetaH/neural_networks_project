#include "neuron.hpp"

float Neuron::eta = 0.15;
float Neuron::alpha = 0.1;

Neuron::Neuron(unsigned numLayerInputs, unsigned numNeuronOutputs, unsigned neuronIndex)
{
    for (unsigned c = 0; c < numNeuronOutputs; ++c)
    {
        m_outWeights.push_back(Connection());
        m_outWeights.back().weight = heWeightInit(numLayerInputs);
    }
    m_myIndex = neuronIndex;
}

float Neuron::heWeightInit(unsigned numLayerInputs) {
    /**
     * @brief He weight initialization
     */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, std::sqrt(2.0 / numLayerInputs));
    return dis(gen);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // For all neurons of the prevoius layer, including bias
    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        Neuron &prev_neuron = prevLayer[i];
        float oldDeltaWeight = prev_neuron.m_outWeights[m_myIndex].deltaweight;

        // - (Learning rate * prev neuron output * gradient) + momentum * old weight change
        float newDeltaWeight =
            -(eta * prev_neuron.getOutputVal() * m_gradient) + alpha * oldDeltaWeight;
        // cout << "using gradient " << m_gradient << endl;

        prev_neuron.m_outWeights[m_myIndex].deltaweight = newDeltaWeight;
        prev_neuron.m_outWeights[m_myIndex].weight += newDeltaWeight;
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
        //sum += m_outWeights[i].weight * nextLayer[i].m_gradient * Neuron::transferFunctionDerivative(nextLayer[i].potential);
    }
    
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    float dow = sumDOW(nextLayer);
    // m_gradient = dow;
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outVal);
}

void Neuron::calcOutputGradients(float targetVal)
{
    float delta = m_outVal - targetVal;
    // m_gradient = delta;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outVal);
}

float Neuron::transferFunction(float x)
{
    return tanh(x);

    // return max(0.0f, x); // ReLU !Doesn't work
}

float Neuron::transferFunctionDerivative(float x)
{
    return 1.0 - x * x; // approximate derivative of tanh

    // return x < 0.0f ? 0.0f : 1.0f; // ReLU !Doesn't work
}

void Neuron::feedForward(const Layer &prevLayer)
{
    float sum = 0.0;

    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        sum += prevLayer[i].getOutputVal() * prevLayer[i].m_outWeights[m_myIndex].weight;
    }
    potential = sum;
    m_outVal = Neuron::transferFunction(sum);
}

void Neuron::setLearningRate(float learningRate)
{
    eta = learningRate;
}

void Neuron::addToAvgBatchGradient()
{
    m_gradient_sum += m_gradient;
    // cout << "add " << m_gradient << endl;
    // cout << "sum " << m_gradient_sum << endl;
}

float Neuron::setAvgGradient(unsigned int batchSize)
{
    m_gradient = m_gradient_sum / batchSize;
    // cout << "avg " << m_gradient << endl;
    return m_gradient;
}

void Neuron::resetGradientSum(){
    m_gradient_sum = 0;
}