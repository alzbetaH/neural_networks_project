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
    // vector<vector<float>> getBatch(); // TODO?

    inline unsigned length() { return m_data.size(); }

private:
    const string m_filepath;
    const float m_divisor; // To normalize the data, divide it by this number when reading

    unsigned m_actIndex;
    vector<vector<float>> m_data;
};
