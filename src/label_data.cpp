/**
 * @file label_data.cpp
 * @brief Implementation of the labelData class for a simple neural network.
 */

#include "label_data.hpp"

LabelData::LabelData(const string filepath, unsigned categories, bool onehot_encoded) :
    m_filepath{filepath},
        m_categories{categories},
        m_onehot_encoded{onehot_encoded},
        m_actIndex{0},
        m_actIndexTrain{0},
        m_actIndexValid{0}
{
    this->readData();
}

void LabelData::readData()
{
    ifstream file(m_filepath.c_str());
    if(!file.is_open())
    {
        throw runtime_error("Unable to open file: " + m_filepath);
    }

    string tmp_line;
    vector<double> tmp_data;
    while(getline(file, tmp_line))
    {
        tmp_data.clear();

        if(m_onehot_encoded) // If the labels are already one-hot encoded
        {
            stringstream ss(tmp_line);

            string tmp_val;
            while(getline(ss, tmp_val, ','))
            {
                tmp_data.push_back(stof(tmp_val));
            }

            // Number of bits should be the same as the number of categories
            assert(m_categories == tmp_data.size());

            m_data.push_back(tmp_data);
        }
        else // If the labels are integers (category numbers)
        {
            unsigned label = stof(tmp_line);
            vector<double> label_onehot = this->onehotEncode(label);
            m_data.push_back(label_onehot);
        }
    }

    file.close();
}

void LabelData::splitData(double percentage)
{
    const size_t splitIndex = static_cast<size_t>(percentage * m_data.size());

    // Split the data into training and validation sets
    m_trainingData.assign(m_data.begin(), m_data.begin() + splitIndex);
    m_validationData.assign(m_data.begin() + splitIndex, m_data.end());
}

vector<double> LabelData::onehotEncode(unsigned label)
{
    if (label >= m_categories) {
        throw std::out_of_range("Cannot one-hot encode label" + to_string(label));
    }

    vector<double> encoded(m_categories, 0.0);
    encoded[label] = 1;
    return encoded;
}

unsigned LabelData::onehotDecode(const std::vector<double>& encoded) {
    if (encoded.size() != m_categories) {
        throw std::runtime_error("Invalid one-hot encoded vector size");
    }

    // Find the iterator pointing to the maximum element in the encoded vector.
    auto max_it = std::max_element(encoded.begin(), encoded.end());

    // Check if the maximum element is 1 (as expected in one-hot encoding).
    if (*max_it != 1.0) {
        throw std::runtime_error("Invalid one-hot encoded vector");
    }

    // Calculate and return the index of the maximum element, which is the original label.
    return static_cast<unsigned>(distance(encoded.begin(), max_it));
}

void LabelData::shuffleData(unsigned seed)
{
    shuffle(m_trainingData.begin(), m_trainingData.end(), default_random_engine(seed));
}

vector<double> &LabelData::getNext()
{
    int idx_to_ret = m_actIndex;
    m_actIndex++;
    if(m_actIndex == m_data.size())
    {
        m_actIndex = 0;
    }
    return m_data[idx_to_ret];
}

vector<double> &LabelData::getNextTrain()
{
    int idx_to_ret = m_actIndexTrain;
    m_actIndexTrain++;
    if(m_actIndexTrain == m_trainingData.size())
    {
        m_actIndexTrain = 0;
    }
    return m_trainingData[idx_to_ret];
}

vector<double> &LabelData::getNextValid()
{
    int idx_to_ret = m_actIndexValid;
    m_actIndexValid++;
    if(m_actIndexValid == m_validationData.size())
    {
        m_actIndexValid = 0;
    }
    return m_validationData[idx_to_ret];
}