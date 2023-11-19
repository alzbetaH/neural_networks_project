#include "input_data.hpp"

InputData::InputData(const string filepath, const float divisor, const unsigned batchSize) :
        m_filepath{filepath},
        m_divisor{divisor},
        m_actIndex{0},
        m_actIndexTrain{0},
        m_actIndexValid{0},
        m_batchIndex{0},
        m_batchSize{batchSize}
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

void InputData::splitData(float percentage)
{
    const size_t splitIndex = static_cast<size_t>(percentage * m_data.size());

    // Split the data into training and validation sets
    m_trainingData.assign(m_data.begin(), m_data.begin() + splitIndex);
    m_validationData.assign(m_data.begin() + splitIndex, m_data.end());
}

void InputData::shuffleData(unsigned seed)
{
    shuffle(m_trainingData.begin(), m_trainingData.end(), default_random_engine(seed));
}

vector<float> &InputData::getNext()
{
    int idx_to_ret = m_actIndex;
    m_actIndex++;
    if(m_actIndex == m_data.size())
    {
        m_actIndex = 0;
    }

    return m_data[idx_to_ret];
}

vector<float> &InputData::getNextTrain()
{
    int idx_to_ret = m_actIndexTrain;
    m_actIndexTrain++;
    if(m_actIndexTrain == m_trainingData.size())
    {
        m_actIndexTrain = 0;
    }

    return m_trainingData[idx_to_ret];
}

vector<float> &InputData::getNextValid()
{
    int idx_to_ret = m_actIndexValid;
    m_actIndexValid++;
    if(m_actIndexValid == m_validationData.size())
    {
        m_actIndexValid = 0;
    }

    return m_validationData[idx_to_ret];
}

unsigned InputData::getNextBatchSize()
{
    /**
     * @brief Returns the batch size of the next batch - desired batch size if
     * possible, otherwise the number of remaining samples
     */
    return min(m_batchSize, static_cast<unsigned>(m_trainingData.size() - m_actIndexTrain));
}
