#include "neuron.hpp"

float Neuron::eta = 0.15;
float Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numLayerInputs, unsigned numNeuronOutputs, unsigned neuronIndex)
{
    for (unsigned c = 0; c < numNeuronOutputs; ++c)
    {
        m_outWeights.push_back(heWeightInit(numLayerInputs));
        m_outWeightsDeltas.push_back(0.0);
        m_outWeightsGradients.push_back(0.0);
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

void Neuron::updateWeights()
{
    for (unsigned i = 0; i < m_outWeights.size(); ++i)
    {
        float oldDeltaWeight = m_outWeightsDeltas[i];

        // - (Learning rate * prev neuron output * gradient) + momentum * old weight change
        float newDeltaWeight =
            -(eta * m_outWeightsGradients[i]) + alpha * oldDeltaWeight;
        // cout << "using gradient " << m_gradient << endl;

        m_outWeightsDeltas[i] = newDeltaWeight;
        m_outWeights[i] += newDeltaWeight;
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
        sum += m_outWeights[i] * nextLayer[i].m_gradient;
        // sum += m_outWeights[i] * nextLayer[i].m_gradient * Neuron::transferFunctionDerivative(nextLayer[i].m_potential);
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    /**
     * @brief m_gradient = transfer_function_derivative(inner_potential or output_value) * (sum_for_neurons_r_in_next_layer) (weight_from_this_neuron_to_r * gradient_of_r)
     */
    // m_gradient = sumDOW(nextLayer) * Neuron::transferFunctionDerivative(m_outVal);
    m_gradient = sumDOW(nextLayer) * Neuron::transferFunctionDerivative(m_potential);
}

void Neuron::calcOutputGradients(float targetVal)
{
    /**
     * @brief m_gradient = transfer_function_derivative(inner_potential or output_value) * (target_value - output_value)
     */
    m_gradient = (m_outVal - targetVal) * Neuron::transferFunctionDerivative(m_outVal); // for sigmoid or tanh, use the output value
    // m_gradient = (m_outVal - targetVal) * Neuron::transferFunctionDerivative(m_potential); // for relu, use the inner potential? Doesn't really work
}

void Neuron::calcWeightGradients(const Layer &nextLayer)
{
    for(unsigned i = 0; i < m_outWeights.size(); ++i)
    {
        // m_outWeightsGradients[i] = m_gradient * nextLayer[i].getOutputVal();
        m_outWeightsGradients[i] += nextLayer[i].m_gradient * m_outVal;
    }
}

float Neuron::transferFunction(float x)
{
    // return tanh(x);

    return max(0.0f, x); // ReLU !Doesn't work
}

float Neuron::transferFunctionDerivative(float x)
{
    // return 1.0 - x * x; // approximate derivative of tanh

    return x < 0.0f ? 0.0f : 1.0f; // ReLU !Doesn't work
}

void Neuron::feedForward(const Layer &prevLayer)
{
    m_potential = 0.0;
    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        m_potential += prevLayer[i].getOutputVal() * prevLayer[i].m_outWeights[m_myIndex];
    }

    m_outVal = Neuron::transferFunction(m_potential);
}

void Neuron::setLearningRate(float learningRate)
{
    eta = learningRate;
}

void Neuron::calcAvgGradient(unsigned int batchSize)
{
    for(unsigned i = 0; i < m_outWeightsGradients.size(); ++i)
    {
        m_outWeightsGradients[i] /= batchSize;
    }
}

void Neuron::resetGradientSum(){
    for(unsigned i = 0; i < m_outWeightsGradients.size(); ++i)
    {
        m_outWeightsGradients[i] = 0.0;
    }
}