#include "input_data.hpp"

InputData::InputData(const string filepath, const float divisor) :
        m_filepath{filepath},
        m_divisor{divisor},
        m_actIndex{0},
        m_batchIndex{0}
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

void InputData::getBatch(int batch_size)
{
    /**
     * @brief Prepare a new batch - vector of input samples. A batch can be
     * smaller than the batch_size if the end of data is reached
     */
    m_batch.clear();
    for (int i = 0; i < batch_size; ++i)
    {
        m_batch.push_back(m_data[m_actIndex + i]);

        if (m_actIndex + i + 1 == m_data.size())
        {
            break;
        }
    }
    m_act_batch_size = m_batch.size();
}

vector<float> &InputData::getNextInBatch(unsigned batch_size)
{
    m_actIndex++;
    if(m_actIndex == m_data.size())
    {
        m_actIndex = 0;
    }

    int batch_idx_to_ret = m_batchIndex;
    m_batchIndex++;
    if((m_batchIndex == batch_size) || (m_actIndex == 0))
    {
        m_batchIndex = 0;
    }

    return m_batch[batch_idx_to_ret];
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

int InputData::getActualBatchSize()
{
    return m_act_batch_size;
}
