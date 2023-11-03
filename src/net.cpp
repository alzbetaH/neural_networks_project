#include "net.hpp"
#include <cassert>
#include <limits>

float Net::m_recentAverageSmoothingFactor = 100.0;

Net::Net(const vector<unsigned> &topology)
{

    unsigned numLayers = topology.size();

    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());

        //if last layer no outputs are set
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        
        //creating neurons according to given topology
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(topology[layerNum], numOutputs, neuronNum));
        }

        //initial bias value
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

    //Huber loss
    m_error = 0.0;
    for (unsigned i = 0; i < outputLayer.size() - 1; ++i)
    {
        float delta = targetVals[i] - outputLayer[i].getOutputVal();
        m_error += delta * delta;
    }
    m_error = m_error / 2;

    // // Root mean squared error
    // m_error = 0.0;
    // for (unsigned i = 0; i < outputLayer.size() - 1; ++i)
    // {
    //     float delta = targetVals[i] - outputLayer[i].getOutputVal();
    //     m_error += delta * delta;
    // }
    // m_error /= outputLayer.size() - 1;
    // m_error = sqrt(m_error);

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
    for (unsigned i = 0; i < outputLayer.size() -1; ++i)
    {
        outputLayer[i].calcOutputGradients(targetVals[i]);
        // outputLayer[i].addToAvgBatchGradient();
    }

    //gradients on hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0 ; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned i = 0; i < hiddenLayer.size(); ++i)
        {
            hiddenLayer[i].calcHiddenGradients(nextLayer);
            // outputLayer[i].addToAvgBatchGradient();
        }
        
    }

}

void Net::updateWeights()
{
    //for all layers from outputs to first hidden layer update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned i = 0; i < layer.size() - 1; ++i)
        {
            layer[i].updateInputWeights(prevLayer);
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

void Net::setAvgGradient(unsigned int batchSize)
{
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0 ; --layerNum)
    {
        Layer &actLayer = m_layers[layerNum];

        for (unsigned i = 0; i < actLayer.size(); ++i)
        {
            actLayer[i].setAvgGradient(batchSize + 1);
        }
        
    }
}