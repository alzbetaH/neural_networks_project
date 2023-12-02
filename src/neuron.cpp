#include "neuron.hpp"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.9;
double Neuron::decay = 0.9;
double Neuron::epsilon = 1e-8;

Neuron::Neuron(unsigned numLayerInputs, unsigned numNeuronOutputs, unsigned neuronIndex)
{
    for (unsigned c = 0; c < numNeuronOutputs; ++c)
    {
        m_outWeights.push_back(heWeightInit(numLayerInputs));
        m_outWeightsDeltas.push_back(0.0);
        m_outWeightsGradients.push_back(0.0);
    }
    m_myIndex = neuronIndex;
    dropout_probability = 0;
}

double Neuron::heWeightInit(unsigned numLayerInputs) {
    /**
     * @brief He weight initialization
     */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, std::sqrt(2.0 / numLayerInputs));
    return dis(gen);
}

void Neuron::setDropout(double probability)
{
    dropout_probability = probability;
}

// void Neuron::updateWeights()
// {
//     for (unsigned i = 0; i < m_outWeights.size(); ++i)
//     {
//         double oldDeltaWeight = m_outWeightsDeltas[i];

//         // - (Learning rate * prev neuron output * gradient) + momentum * old weight change
//         double newDeltaWeight =
//             -(eta * m_outWeightsGradients[i]) + alpha * oldDeltaWeight;
//         // outWeightsGratients -> nans
//         // if (to_string(newDeltaWeight) == "-nan")
//         // {
//         //     for (unsigned i = 0; i < m_outWeights.size(); ++i)
//         //     {
//         //         cout << "out_weight: "<< m_outWeightsGradients[i] << " old delta: " << oldDeltaWeight << endl;
//         //     }
            
//         //     exit(0);
//         // }
//         newDeltaWeight = abs(newDeltaWeight) < 1e-14 ? 0.0 : newDeltaWeight;
//         m_outWeightsDeltas[i] = newDeltaWeight;
//         m_outWeights[i] += newDeltaWeight;
//     }
// }

// RMSprop, learning rate set to 0.001
void Neuron::updateWeights()
{
    for (unsigned i = 0; i < m_outWeights.size(); ++i)
    {
        double gradient = m_outWeightsGradients[i];

        m_outWeightsDeltas[i] = decay * m_outWeightsDeltas[i] + (1 - decay) * gradient * gradient;

        m_outWeights[i] += -(eta / (sqrt(m_outWeightsDeltas[i]) + epsilon)) * gradient;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    /**
     * @brief Sum of derivatives of weights
     */
    double sum = 0.0;

    for (unsigned i = 0; i < nextLayer.size() - 1; ++i)
    {
        sum += m_outWeights[i] * nextLayer[i].m_gradient;
    }
    // if (to_string(sum) == "-nan")
    // {
    //     for (unsigned i = 0; i < nextLayer.size() - 1; ++i)
    //     {
    //         cout << "weight "<< m_outWeights[i] << " m_gradient " << nextLayer[i].m_gradient << endl;
    //     }
        
    //     exit(0);
    // }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    /**
     * @brief m_gradient = transfer_function_derivative(inner_potential or output_value) * (sum_for_neurons_r_in_next_layer) (weight_from_this_neuron_to_r * gradient_of_r)
     */
    // m_gradient = sumDOW(nextLayer) * Neuron::transferFunctionDerivative(m_outVal); // For tanh or sigmoid, use the output value
    m_gradient = sumDOW(nextLayer) * Neuron::transferFunctionDerivative(m_potential); // For relu, use the inner potential
    // if (to_string(m_gradient) == "-nan")
    // {
    //     cout << "sumDOW "<< sumDOW(nextLayer) << " derivative " << Neuron::transferFunctionDerivative(m_potential) << endl;
    //     exit(0);
    // }
    
}

void Neuron::calcOutputGradients(double targetVal)
{
    /**
     * @brief m_gradient = transfer_function_derivative(inner_potential or output_value) * (target_value - output_value)
     */
    // m_gradient = (m_outVal - targetVal) * Neuron::transferFunctionDerivative(m_outVal);
    m_gradient = m_outVal - targetVal; // For softmax
    // if (to_string(m_gradient) == "-nan")
    // {
    //     cout << "outVal "<< m_outVal << " targetval " << targetVal << endl;
    // }
    // if (to_string(m_gradient) == "-nan")
    // {
    //     cout << "outVal "<< m_outVal << " targetval " << targetVal << endl;
    // }
}

void Neuron::calcWeightGradients(const Layer &nextLayer)
{
    for(unsigned i = 0; i < m_outWeights.size(); ++i)
    {
        // m_outWeightsGradients[i] = m_gradient * nextLayer[i].getOutputVal();
        m_outWeightsGradients[i] += nextLayer[i].m_gradient * m_outVal;
    }
    // if(to_string(m_outWeightsGradients[m_outWeights.size() - 1]) == "-nan")
    // {
    //     for(unsigned i = 0; i < m_outWeights.size(); ++i)
    //     {
    //         cout << "gradient "<< nextLayer[i].m_gradient << " outVal " << m_outVal << endl;
    //     }
    //     exit(0);
    // }
    
}

double Neuron::transferFunction(double x)
{
    // if (to_string(x) == "-nan")
    // {
    //     cout << "x = "<< x << endl;
    //     exit(0);
    // }
    return max(0.0, x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return x < 0.0f ? 0.0f : 1.0f;
}

void Neuron::calcPotential(const Layer &prevLayer)
{
    m_potential = 0.0;

    // Apply dropout with probability
    if((rand() / static_cast<float>(RAND_MAX)) < dropout_probability){
        return;
    }

    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        m_potential += prevLayer[i].getOutputVal() * prevLayer[i].m_outWeights[m_myIndex];
    }
    m_potential = abs(m_potential) < 1e-14 ? 0.0 : m_potential;
    // if (to_string(m_potential) == "-nan")
    // {
    //     cout << "nan by calculatin potential" << endl;
    //     for (unsigned i = 0; i < prevLayer.size(); ++i)
    //     {
    //         cout << prevLayer[i].getOutputVal() << " " << prevLayer[i].m_outWeights[m_myIndex] << endl;
    //     }
    //     exit(0);
    // }
}

void Neuron::calcOutput()
{
    m_outVal = Neuron::transferFunction(m_potential);
}

void Neuron::setLearningRate(double learningRate)
{
    eta = learningRate;
}

void Neuron::calcAvgGradient(unsigned int batchSize)
{
    for(unsigned i = 0; i < m_outWeightsGradients.size(); ++i)
    {
        m_outWeightsGradients[i] /= batchSize;
        // if(to_string(m_outWeightsGradients[i]) == "-nan")
        // {
        //     cout << "calc potential" << endl;
        //     cout << "output "<< m_outWeightsGradients[i] << " batch " << batchSize << endl;
        //     exit(0);
        // }
    }
}

void Neuron::resetGradientSum(){
    for(unsigned i = 0; i < m_outWeightsGradients.size(); ++i)
    {
        m_outWeightsGradients[i] = 0.0;
    }
}