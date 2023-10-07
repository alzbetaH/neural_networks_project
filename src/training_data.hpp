#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>


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

