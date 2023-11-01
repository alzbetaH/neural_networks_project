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
    InputData(const string filepath, const float divisor);

    void readData();
    void shuffleData(unsigned seed);
    vector<float> &getNext();
    void getBatch(int batch_size); // TODO?
    vector<float> &getNextInBatch(int batch_size);
    int getActualBatchSize();

    inline unsigned length() { return m_data.size(); }

private:
    const string m_filepath;
    const float m_divisor; // To normalize the data, divide it by this number when reading

    unsigned m_actIndex;
    unsigned m_batchIndex;
    unsigned act_batch_size;
    vector<vector<float>> m_data;
    vector<vector<float>> mini_batch;
};
