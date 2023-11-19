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
    InputData(const string filepath, const float divisor, unsigned batchSize);

    void readData();
    void splitData(float percentage);
    void shuffleData(unsigned seed);
    vector<float> &getNext();
    unsigned getNextBatchSize();
    vector<float> &getNextTrain();
    vector<float> &getNextValid();


    inline unsigned length() { return m_data.size(); }
    vector<vector<float>> m_data;
    vector<vector<float>> m_trainingData;
    vector<vector<float>> m_validationData;


private:
    const string m_filepath;
    const float m_divisor; // To normalize the data, divide it by this number when reading

    unsigned m_actIndex;
    unsigned m_actIndexTrain;
    unsigned m_actIndexValid;
    unsigned m_batchIndex;
    unsigned m_act_batch_size;
    const unsigned m_batchSize;
    
    // vector<vector<float>> m_data;
    // vector<vector<float>> m_trainingData;
    // vector<vector<float>> m_validationData;

    vector<vector<float>> m_batch;
};
