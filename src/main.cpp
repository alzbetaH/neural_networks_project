#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);
private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);

    //strem class operating with strings, read until whitespace
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0)
    {
        cout << "Error by getting topology" << endl;
        abort();
    }

    while (!ss.eof())
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    
    return;
}

TrainingData::TrainingData(const string filename)
{
    //c_str() converts string into array
    m_trainingDataFile.open(filename.c_str());
};

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
        {
            inputVals.push_back(oneValue);
        }
        
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
        {
            targetOutputVals.push_back(oneValue);
        }
        
    }
    return targetOutputVals.size();
}

struct Connection
{
    double weight;
    double deltaweight;
};

class Neuron;

typedef vector<Neuron> Layer;


class Neuron
{
private:
    static double eta; //[0.0...1.0] overrall learning rate
    static double alpha; //[0.0...n] multiplier of last weight change (momentum) 
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outVal;
    vector<Connection> m_outWeights;
    unsigned m_myIndex;
    double m_gradient;

public:
    Neuron(unsigned numOutVals, unsigned myIndex);
    void setOutputVal(double val) { m_outVal = val;}
    double getOutputVal(void) const { return m_outVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
};

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


class Net
{
private:
    vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;

public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const {return m_recentAverageError;};
};

double Net::m_recentAverageSmoothingFactor = 100.0;

void Net::getResults(vector<double> &resultVals) const 
{
    resultVals.clear();

    for (unsigned i = 0; i < m_layers.back().size() - 1; ++i)
    {
        //cout << "out: " << m_layers.back()[i].getOutputVal() << endl;
        resultVals.push_back(m_layers.back()[i].getOutputVal());
    }
    
}

void Net::backProp(const vector<double> &targetVals) 
{
    //mean square error
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned i = 0; i < outputLayer.size() - 1; ++i)
    {
        double delta = targetVals[i] - outputLayer[i].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

    //gradients for output neurons
    for (unsigned i = 0; i < outputLayer.size() -1; ++i)
    {
        outputLayer[i].calcOutputGradients(targetVals[i]);
    }

    //gradients on hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0 ; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned i = 0; i < hiddenLayer.size(); ++i)
        {
            hiddenLayer[i].calcHiddenGradients(nextLayer);
        }
        
    }
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

void Net::feedForward(const vector<double> &inputVals) 
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

Net::Net(const vector<unsigned> &topology)
{

    unsigned numLayers = topology.size();

    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());

        //if last layer no outputs are set
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        
        //creating neurons according to given topology
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
        }

        //initial bias value
        m_layers.back().back().setOutputVal(1.0);
    }


}

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}

int main(){

    TrainingData trainData("data/trainingSamples.txt");

    vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof())
    {
        trainingPass++;
        cout << endl << "Pass " << trainingPass;

        if (trainData.getNextInputs(inputVals) != topology[0] )
        {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);
        
        myNet.getResults(resultVals);
        //cout << "out: " << m_layers.back()[i].getOutputVal() << endl;
        //cout << "res val: " << resultVals[0] << endl;
        showVectorVals("Outputs:", resultVals);

        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());
    
        myNet.backProp(targetVals);

        cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
    }
    
    cout << endl << "Done" << endl;
}