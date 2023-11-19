#include <vector>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>


using namespace std;

class LabelData
{
public:
    LabelData(const string filepath, unsigned categories, bool onehot_encoded);

    void readData();
    void splitData(float percentage);
    vector<float> onehotEncode(unsigned label);
    unsigned onehotDecode(const std::vector<float>& encoded);
    void shuffleData(unsigned seed);
    vector<float> &getNext();
    vector<float> &getNextValid();
    vector<float> &getNextTrain();
    vector<vector<float>> m_data;
    vector<vector<float>> m_trainingData;
    vector<vector<float>> m_validationData;
    // vector<vector<float>> getBatch(); // TODO?

    inline unsigned length() { return m_data.size(); }

private:
    const string m_filepath;
    const unsigned m_categories;
    const bool m_onehot_encoded;

    unsigned m_actIndex;
    unsigned m_actIndexTrain;
    unsigned m_actIndexValid;
    // vector<vector<float>> m_data;
    // vector<vector<float>> m_trainingData;
    // vector<vector<float>> m_validationData;
};
