#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>


using namespace std;

class InputData
{
public:
    InputData(const string filepath, const double divisor, unsigned batchSize);

    void readData();
    void splitData(double percentage);
    void shuffleData(unsigned seed);
    vector<double> &getNext();
    unsigned getNextBatchSize();
    vector<double> &getNextTrain();
    vector<double> &getNextValid();


    inline unsigned length() { return m_data.size(); }
    vector<vector<double>> m_data;
    vector<vector<double>> m_trainingData;
    vector<vector<double>> m_validationData;


private:
    const string m_filepath;
    const double m_divisor; // To normalize the data, divide it by this number when reading

    unsigned m_actIndex;
    unsigned m_actIndexTrain;
    unsigned m_actIndexValid;
    unsigned m_batchIndex;
    unsigned m_act_batch_size;
    const unsigned m_batchSize;
    
    // vector<vector<double>> m_data;
    // vector<vector<double>> m_trainingData;
    // vector<vector<double>> m_validationData;

    vector<vector<double>> m_batch;
};
