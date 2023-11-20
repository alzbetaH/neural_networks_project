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
    void splitData(double percentage);
    vector<double> onehotEncode(unsigned label);
    unsigned onehotDecode(const std::vector<double>& encoded);
    void shuffleData(unsigned seed);
    vector<double> &getNext();
    vector<double> &getNextValid();
    vector<double> &getNextTrain();
    vector<vector<double>> m_data;
    vector<vector<double>> m_trainingData;
    vector<vector<double>> m_validationData;
    // vector<vector<double>> getBatch(); // TODO?

    inline unsigned length() { return m_data.size(); }

private:
    const string m_filepath;
    const unsigned m_categories;
    const bool m_onehot_encoded;

    unsigned m_actIndex;
    unsigned m_actIndexTrain;
    unsigned m_actIndexValid;
    // vector<vector<double>> m_data;
    // vector<vector<double>> m_trainingData;
    // vector<vector<double>> m_validationData;
};
