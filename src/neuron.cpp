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
    // For all inputs - neurons of the prevoius layer, including bias
    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        Neuron &input_neuron = prevLayer[i];
        float oldDeltaWeight = input_neuron.m_outWeights[m_myIndex].deltaweight;

        // - (Learning rate * prev neuron output * gradient) + momentum * old weight change
        float newDeltaWeight =
            -(eta * input_neuron.getOutputVal() * m_gradient) + alpha * oldDeltaWeight;
        // cout << "using gradient " << m_gradient << endl;

        input_neuron.m_outWeights[m_myIndex].deltaweight = newDeltaWeight;
        input_neuron.m_outWeights[m_myIndex].weight += newDeltaWeight;
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
        //sum += m_outWeights[i].weight * nextLayer[i].m_gradient * Neuron::transferFunctionDerivative(nextLayer[i].m_potential);
    }
    
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    /**
     * @brief m_gradient = transfer_function_derivative(inner_potential or output_value) * (sum_for_neurons_r_in_next_layer) (weight_from_this_neuron_to_r * gradient_of_r)
     */
    m_gradient = sumDOW(nextLayer) * Neuron::transferFunctionDerivative(m_outVal);
    m_gradient_sum += m_gradient;
}

void Neuron::calcOutputGradients(float targetVal)
{
    /**
     * @brief m_gradient = transfer_function_derivative(inner_potential or output_value) * (target_value - output_value)
     */
    m_gradient = (m_outVal - targetVal) * Neuron::transferFunctionDerivative(m_outVal); // for sigmoid or tanh, use the output value
    // m_gradient = (m_outVal - targetVal) * Neuron::transferFunctionDerivative(m_potential); // for relu, use the inner potential? Doesn't really work
    m_gradient_sum += m_gradient;
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
    m_potential = 0.0;
    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        m_potential += prevLayer[i].getOutputVal() * prevLayer[i].m_outWeights[m_myIndex].weight;
    }

    m_outVal = Neuron::transferFunction(m_potential);
}

void Neuron::setLearningRate(float learningRate)
{
    eta = learningRate;
}

void Neuron::calcAvgGradient(unsigned int batchSize)
{
    m_gradient = m_gradient_sum / batchSize;
    // cout << "avg " << m_gradient << endl;
}

void Neuron::resetGradientSum(){
    m_gradient_sum = 0;
}