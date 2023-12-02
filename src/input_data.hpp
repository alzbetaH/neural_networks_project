/**
 * @file input_data.hpp
 * @brief Declaration of the InputData class and its methods.
 */
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>


using namespace std;

/**
 * @class InputData
 * @brief Class for handling input data, providing methods for reading, splitting, and accessing data.
 */
class InputData
{
public:
    /**
     * @brief Constructs for InputData class.
     *
     * @param filepath The path to the file containing input data.
     * @param divisor The divisor applied to each input value.
     * @param batchSize The desired batch size for training.
     */
    InputData(const string filepath, const double divisor, unsigned batchSize);

    /**
     * @brief Reads input data from a file and stores it in the object.
     */
    void readData();

    /**
     * @brief Split the labeled data into training and validation sets based on a given percentage.
     * 
     * @param percentage The percentage of data to be used for training.
     */
    void splitData(double percentage);

    /**
     * @brief Shuffle the training data using a provided seed.
     * 
     * @param seed The seed for the random number generator used for shuffling.
     */
    void shuffleData(unsigned seed);

    /**
     * @brief Get the next data input from the entire dataset.
     * 
     * @return A reference to the next data input in the entire dataset.
     */
    vector<double> &getNext();

    /**
     * @brief Get the next data input from the training set.
     * 
     * @return A reference to the next data input in the training set.
     */
    vector<double> &getNextTrain();

    /**
     * @brief Get the next data input from the validation set.
     * 
     * @return A reference to the next data input in the validation set.
     */
    vector<double> &getNextValid();

    /**
     * @brief Retrieves the batch size for the next training batch.
     *
     * @return The batch size, either the desired batch size or the remaining samples.
     */
    unsigned getNextBatchSize();

    /**
     * @brief Resets the internal indices for accessing data.
     */
    inline void resetIndex(){ m_actIndex = 0; m_actIndexTrain = 0; m_actIndexValid = 0; m_batchIndex = 0; }

    /**
     * @brief Gets the total number of data inputs.
     * 
     * @return The total number of data inputs in the dataset.
     */
    inline unsigned length() { return m_data.size(); }

    /**
     * @brief Gets the number of data inputs in the training set.
     * 
     * @return The number of data inputs in the training set.
     */
    inline unsigned validLength() { return m_validationData.size(); };

    /**
     * @brief Gets the number of data inputs in the validation set.
     * 
     * @return The number of data inputs in the validation set.
     */
    inline unsigned trainLength() { return m_trainingData.size(); }

private:
    /**
     * @brief Path to the file containing input data.
     */
    const string m_filepath;

    /**
     * @brief Divisor to normalize the data by dividing each input value.
     */
    const double m_divisor; 

    /**
     * @brief Index for accessing the overall labeled data.
     */
    unsigned m_actIndex;

    /**
     * @brief Index for accessing the training set data.
     */
    unsigned m_actIndexTrain;

    /**
     * @brief Index for accessing the validation set data.
     */
    unsigned m_actIndexValid;

    /**
     * @brief Index for tracking the current batch in the training set.
     */
    unsigned m_batchIndex;

    /**
     * @brief Number of remaining samples in the current batch.
     */
    unsigned m_act_batch_size;

    /**
     * @brief Desired batch size for training.
     */
    const unsigned m_batchSize;

    /**
     * @brief Matrix containing all input data.
     */
    vector<vector<double>> m_data;

    /**
     * @brief Matrix containing input data for training.
     */
    vector<vector<double>> m_trainingData;

    /**
     * @brief Matrix containing input data for validation.
     */
    vector<vector<double>> m_validationData;

    /**
     * @brief Matrix containing the current batch of input data.
     */
    vector<vector<double>> m_batch;
};
