#include "net.hpp"
#include <cassert>
#include <limits>

float Net::m_recentAverageSmoothingFactor = 100.0;

Net::Net(const vector<unsigned> &topology) : m_error(0.0), m_recentAverageError(0.0)
{

    unsigned numLayers = topology.size();

    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());

        // if last layer no outputs are set
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // creating neurons according to given topology (one additional for bias)
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(topology[layerNum], numOutputs, neuronNum));
        }

        // bias value
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::getResults(vector<float> &resultVals) const 
{
    resultVals.clear();

    for (unsigned i = 0; i < m_layers.back().size() - 1; ++i)
    {
        //cout << "out: " << m_layers.back()[i].getOutputVal() << endl;
        resultVals.push_back(m_layers.back()[i].getOutputVal());
    }
    
}

void Net::backProp(const vector<float> &targetVals)
{
    Layer &outputLayer = m_layers.back();

    // Mean squared error
    m_error = 0.0;
    for (unsigned i = 0; i < outputLayer.size() - 1; ++i)
    {
        m_error += pow(targetVals[i] - outputLayer[i].getOutputVal(), 2);
    }
    m_error /= outputLayer.size() - 1; // Mean
    m_error /= 2; // Simplify the derivative

    // Root mean squared error
    // m_error = 0.0;
    // for (unsigned i = 0; i < outputLayer.size() - 1; ++i)
    // {
    //     m_error += pow(targetVals[i] - outputLayer[i].getOutputVal(), 2);
    // }
    // m_error /= outputLayer.size() - 1; // Mean
    // m_error /= 2; // Ease the derivative
    // m_error = sqrt(m_error); // Root

    // Cross entropy loss
    // for(unsigned i = 0; i < outputLayer.size() - 1; ++i)
    // {
    //     if(targetVals[i] == 1.0)
    //     {
    //         m_error = -log(outputLayer[i].getOutputVal());
    //     }
    // }

    // Categorical cross entropy loss
    // m_error = 0;
    // for(unsigned i = 0; i < outputLayer.size(); ++i)
    // {
    //     double outputVal = max(numeric_limits<float>::min(), outputLayer[i].getOutputVal()); // Avoid log(0)
    //     m_error -= targetVals[i] * log(outputVal);
    // }

    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

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
}

void Net::updateWeights()
{
    //for all layers from outputs to first hidden layer: update weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned neuron_i = 0; neuron_i < layer.size() - 1; ++neuron_i) // Ignore bias...
        {
            layer[neuron_i].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<float> &inputVals)
{
    //the number of input values is the same as the number of input neurons
    // -1 because of bias
    assert(inputVals.size() == m_layers[0].size() - 1);

    //set values of input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    //cout << "out: " << m_layers[0][0].getOutputVal() << endl;

    //forward propagation
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum -1];
        for (unsigned i = 0; i < m_layers[layerNum].size() - 1; ++i)
        {
            m_layers[layerNum][i].feedForward(prevLayer);
        }
    }
};

void Net::calcAvgGradient(unsigned int batchSize)
{
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0 ; --layerNum)
    // for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &actLayer = m_layers[layerNum];

        for (unsigned i = 0; i < actLayer.size() - 1; ++i)
        {
            actLayer[i].calcAvgGradient(batchSize);
        }
    }
}

void Net::resetGradientSum(){
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0 ; --layerNum)
    // for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &actLayer = m_layers[layerNum];

        for (unsigned i = 0; i < actLayer.size() - 1; ++i)
        {
            actLayer[i].resetGradientSum();
        }
    }
}
