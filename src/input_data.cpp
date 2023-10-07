#include "input_data.hpp"

InputData::InputData(const string filepath, const float divisor) :
        m_filepath{filepath},
        m_divisor{divisor},
        m_actIndex{0}
{
    this->readData();
}

void InputData::readData()
{
    ifstream file(m_filepath.c_str());
    if(!file.is_open())
    {
        throw runtime_error("Unable to open file: " + m_filepath);
    }

    string tmp_line;
    vector<float> tmp_data;
    while(getline(file, tmp_line))
    {
        tmp_data.clear();
        stringstream ss(tmp_line);

        string tmp_val;
        while(getline(ss, tmp_val, ','))
        {
            tmp_data.push_back(stof(tmp_val) / m_divisor);
        }

        m_data.push_back(tmp_data);
    }

    file.close();

}

void InputData::shuffleData(unsigned seed)
{
    shuffle(m_data.begin(), m_data.end(), default_random_engine(seed));
}

vector<float> &InputData::getNext()
{
    m_actIndex++;
    if(m_actIndex == m_data.size())
    {
        m_actIndex = 0;
    }

    return m_data[m_actIndex];
}
