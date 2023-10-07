#include <cassert>
#include <cmath>

#include "main.hpp"


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