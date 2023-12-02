#include "neuron.hpp"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.9;
double Neuron::decay = 0.9;
double Neuron::epsilon = 1e-8;


Neuron::Neuron(unsigned numLayerInputs, unsigned numNeuronOutputs, unsigned neuronIndex, unsigned seed)
    : m_myIndex{neuronIndex},
    dropout_probability{0},
    dropout_probability_int{0}
{
    std::mt19937 generator(seed); // To generate seeds individual to weights

    for (unsigned c = 0; c < numNeuronOutputs; ++c)
    {
        m_outWeights.push_back(heWeightInit(numLayerInputs, generator()));
        m_outWeightsDeltas.push_back(0.0);
        m_outWeightsGradients.push_back(0.0);
    }
    m_myIndex = neuronIndex;
}

void Neuron::setLearningRate(double learningRate)
{
    eta = learningRate;
}

double Neuron::heWeightInit(unsigned numLayerInputs, unsigned seed) 
{
    std::mt19937 generator(seed);
    std::normal_distribution<> distribution(0, std::sqrt(2.0 / numLayerInputs));
    return distribution(generator);
}

void Neuron::setDropout(double probability)
{
    dropout_probability = probability;
    dropout_probability_int = static_cast<unsigned>(probability * RAND_MAX);
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
    double sum = 0.0;

    for (unsigned i = 0; i < nextLayer.size() - 1; ++i)
    {
        sum += m_outWeights[i] * nextLayer[i].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    m_gradient = sumDOW(nextLayer) * Neuron::transferFunctionDerivative(m_potential); 
}

void Neuron::calcOutputGradients(double targetVal)
{
    m_gradient = m_outVal - targetVal; // For softmax
}

void Neuron::calcWeightGradients(const Layer &nextLayer)
{
    for(unsigned i = 0; i < m_outWeights.size(); ++i)
    {
        // m_outWeightsGradients[i] = m_gradient * nextLayer[i].getOutputVal();
        m_outWeightsGradients[i] += nextLayer[i].m_gradient * m_outVal;
    }
}

double Neuron::transferFunction(double x)
{
    return max(0.0, x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return x < 0.0f ? 0.0f : 1.0f;
}

void Neuron::calcPotential(const Layer &prevLayer)
{
    m_potential = 0.0;

    for (unsigned i = 0; i < prevLayer.size(); ++i)
    {
        m_potential += prevLayer[i].getOutputVal() * prevLayer[i].m_outWeights[m_myIndex];
    }
    m_potential = abs(m_potential) < 1e-14 ? 0.0 : m_potential;
}

void Neuron::calcOutput()
{
    // Apply dropout with probability
    if(rand() < dropout_probability_int){
        m_potential = 0.0;
        m_outVal = 0.0;
        return;
    }else{
        // Remember to scale the output value by dropout probability
        // (if probability is 0, nothing happens to the value)
        m_outVal = Neuron::transferFunction(m_potential) / (1 - dropout_probability);
    }
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