/**
 * @file net.cpp
 * @brief Implementation of the Net class for a simple neural network.
 */

#include "net.hpp"
#include <cassert>
#include <limits>
#include <string>

Net::Net(const vector<unsigned> &topology, unsigned seed) : m_error(0.0)
{
    unsigned numLayers = topology.size();

    std::mt19937 generator(seed); // To generate seeds individual to neurons

    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());

        // if last layer no outputs are set
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // creating neurons according to given topology (one additional for bias)
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(topology[layerNum], numOutputs, neuronNum, generator()));
        }
        // bias value
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::getResults(vector<double> &resultVals) const 
{
    resultVals.clear();

    for (unsigned i = 0; i < m_layers.back().size() - 1; ++i)
    {
        //cout << "out: " << m_layers.back()[i].getOutputVal() << endl;
        resultVals.push_back(m_layers.back()[i].getOutputVal());
    }
}

double Net::getLoss(const vector<double> &targetVals)
{
    Layer &outputLayer = m_layers.back();
    double loss = 0;
    for(unsigned i = 0; i < outputLayer.size() - 1; ++i) // exclude bias
    {
        double outputVal = max(outputLayer[i].getOutputVal(), (double)(1.0E-15F)); // Avoid log(0)
        loss -= targetVals[i] * log(outputVal);
    }
    return abs(loss) < 1e-14 ? 0.0 : loss;
}

void Net::backProp(const vector<double> &targetVals)
{
    Layer &outputLayer = m_layers.back();

    // Categorical cross entropy loss
    m_error = 0;
    for(unsigned i = 0; i < outputLayer.size() - 1; ++i) // exclude bias
    {
        double outputVal = max(outputLayer[i].getOutputVal(), 0.000001); // Avoid log(0)
        m_error -= targetVals[i] * log(outputVal);
    }

    //gradients for output neurons
    for (unsigned i = 0; i < outputLayer.size() - 1; ++i) // -1 to skip the bias
    {
        outputLayer[i].calcOutputGradients(targetVals[i]);
    }

    //gradients on hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned i = 0; i < hiddenLayer.size() - 1; ++i)
        {
            hiddenLayer[i].calcHiddenGradients(nextLayer);
        }
    }

    // Gradients with respect to the weights
    for (unsigned layerNum = 0; layerNum < m_layers.size() - 1; ++layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &hiddenLayer = m_layers[layerNum + 1];

        for (unsigned i = 0; i < layer.size() - 1; ++i)
        {
            layer[i].calcWeightGradients(hiddenLayer);
        }
    }
}

void Net::updateWeights()
{
    for (unsigned layerNum = 0; layerNum < m_layers.size() - 1; ++layerNum)
    {
        Layer &actLayer = m_layers[layerNum];

        for (unsigned i = 0; i < actLayer.size(); ++i)
        {
            actLayer[i].updateWeights();
        }
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    //the number of input values is the same as the number of input neurons
    // -1 because of bias
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Set values of input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Calculate output values of hidden neurons
    for (unsigned layerNum = 1; layerNum < m_layers.size() - 1; ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum -1];
        for (unsigned i = 0; i < m_layers[layerNum].size() - 1; ++i)
        {
            m_layers[layerNum][i].calcPotential(prevLayer);
            m_layers[layerNum][i].calcOutput();
        }
    }

    // Calculate the network outputs - use softmax
    Layer &outLayer = m_layers[m_layers.size() - 1];
    Layer &prevLayer = m_layers[m_layers.size() - 2];
    double exp_sum = 0.0;
    double val = 0.0;
    for (unsigned i = 0; i < outLayer.size() - 1; ++i)
    {
        outLayer[i].calcPotential(prevLayer);
        val = exp(outLayer[i].getPotential());
        exp_sum += val;
    }

    for (unsigned i = 0; i < outLayer.size() - 1; ++i)
    {
        outLayer[i].setOutputVal(exp(outLayer[i].getPotential()) / exp_sum);
    }
};

void Net::calcAvgGradient(unsigned int batchSize)
{
    for (unsigned layerNum = 0; layerNum < m_layers.size() - 1; ++layerNum)
    {
        Layer &actLayer = m_layers[layerNum];

        for (unsigned i = 0; i < actLayer.size(); ++i)
        {
            actLayer[i].calcAvgGradient(batchSize);
        }
    }
}

void Net::resetGradientSum(){
    for (unsigned layerNum = 0; layerNum < m_layers.size() - 1; ++layerNum)
    {
        Layer &actLayer = m_layers[layerNum];

        for (unsigned i = 0; i < actLayer.size(); ++i)
        {
            actLayer[i].resetGradientSum();
        }
    }
}

int Net::compare_result(const vector<double> &output, const vector<double> &label)
{
    auto maxElementIter = max_element(output.begin(), output.end());
    unsigned index_o = distance(output.begin(), maxElementIter);

    auto maxElementIter_l = max_element(label.begin(), label.end());
    unsigned index_l = distance(label.begin(), maxElementIter_l);

    return (index_o == index_l);
}

void Net::setDropout(unsigned int layer_num, double probability)
{
    Layer &actLayer = m_layers[layer_num];
    for (unsigned i = 0; i < actLayer.size(); ++i)
    {
        actLayer[i].setDropout(probability);
    }
}
