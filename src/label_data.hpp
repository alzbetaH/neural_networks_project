/**
 * @file label_data.hpp
 * @brief Declaration of the labelData class and its methods.
 */

#include <vector>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>


using namespace std;

/**
 * @class LabelData
 * @brief Class for handling labeled data, providing methods for reading, splitting, and accessing data.
 */
class LabelData
{
public:
    /**
     * @brief Constructor for LabelData class.
     * 
     * @param filepath Path to the file containing labeled data.
     * @param categories Number of categories or classes in the labeled data.
     * @param onehot_encoded Flag indicating whether the labels are already one-hot encoded.
     */
    LabelData(const string filepath, unsigned categories, bool onehot_encoded);

    /**
     * @brief Read labeled data from the specified file and store it internally.
     */
    void readData();

    /**
     * @brief Split the labeled data into training and validation sets based on a given percentage.
     * 
     * @param percentage The percentage of data to be used for training.
     */
    void splitData(double percentage);

    /**
     * @brief One-hot encode a label.
     * 
     * @param label The original label to be one-hot encoded.
     * @return The one-hot encoded vector corresponding to the label.
     */
    vector<double> onehotEncode(unsigned label);

    /**
     * @brief Decode a one-hot encoded vector to retrieve the original label.
     * 
     * @param encoded The one-hot encoded vector.
     * @return The original label decoded from the one-hot encoded vector.
     */
    unsigned onehotDecode(const std::vector<double>& encoded);

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
    vector<double> &getNextValid();
    
    /**
     * @brief Get the next data input from the validation set.
     * 
     * @return A reference to the next data input in the validation set.
     */
    vector<double> &getNextTrain();

    /**
     * @brief Resets the internal indices for accessing data.
     */
    inline void resetIndex(){ m_actIndex = 0; m_actIndexTrain = 0; m_actIndexValid = 0; }

    /**
     * @brief Gets the total number of data inputs.
     * 
     * @return The total number of data inputs in the dataset.
     */
    inline unsigned length() { return m_data.size(); };

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
     * @brief Path to the file containing labeled data.
     */
    const string m_filepath;

    /**
     * @brief Number of categories or classes in the labeled data.
     */
    const unsigned m_categories;

    /**
     * @brief Flag indicating whether the labels are already one-hot encoded.
     */
    const bool m_onehot_encoded;

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
     * @brief Container for storing labeled data.
     */
    vector<vector<double>> m_data;

    /**
     * @brief Container for storing training set data.
     */
    vector<vector<double>> m_trainingData;

    /**
     * @brief Container for storing validation set data.
     */
    vector<vector<double>> m_validationData;
};
