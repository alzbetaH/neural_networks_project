/**
 * @file net.hpp
 * @brief Declaration of the Net class and its methods.
 */
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "neuron.hpp"

using namespace std;

/**
 * @class Net
 * @brief Represents a whole net topology, providing methods for training neural network.
 */
class Net
{
private:
    /**
     * @brief Vector holding the layers of neurons in the neural network.
     */
    vector<Layer> m_layers; //m_layers[layerNum][neuronNum]

    /**
     * @brief Current error value of the neural network (cathegorical cross-entropy).
     */
    double m_error;

public:
    /**
     * @brief Constructor for the Net class.
     * 
     * Initializes the neural network with the given topology and seed for random number generation.
     * 
     * @param topology Vector representing the number of neurons in each layer.
     * @param seed Seed for random number generation.
     */
    Net(const vector<unsigned> &topology, unsigned seed);

    /**
     * @brief Get the results (output values) of the neural network.
     * @param resultVals Vector to store the output values.
     */
    void getResults(vector<double> &resultVals) const;

    /**
     * @brief Calculate the loss (categorical cross-entropy) between the network output and target values.
     * @param targetVals Target values for the output layer.
     * @return Loss value.
     */
    double getLoss(const vector<double> &targetVals);

    /**
     * @brief Backpropagate the error and update the network weights.
     * 
     * Computes the gradients and updates the weights of the network using backpropagation.
     * 
     * @param targetVals Target values for the output layer.
     */
    void backProp(const vector<double> &targetVals);


    /**
     * @brief Update the weights of the neural network based on calculated gradients.
     * 
     * Iterates through each layer of the network and updates the weights of each neuron
     * based on the computed gradients during backpropagation.
     */
    void updateWeights();

    /**
     * @brief Perform a feedforward pass to compute the network output.
     * 
     * Sets the input values, calculates the potential and output values of each neuron
     * in hidden layers, and computes the softmax activation for the output layer.
     * 
     * @param inputVals Vector containing the input values to the network.
     */
    void feedForward(const vector<double> &inputVals);

    /**
     * @brief Calculate the average gradient of each weight in the network.
     * 
     * Calculates the average gradient for each weight in the network over a mini-batch.
     * 
     * @param batchSize Number of training examples in the mini-batch.
     */
    void calcAvgGradient(unsigned int batchSize);

    /**
     * @brief Reset the gradient sum for each weight in the network.
     * 
     * Resets the accumulated gradient sum for each weight in the network.
     */
    void resetGradientSum();

    /**
     * @brief Compare the predicted output with the ground truth label.
     * 
     * Compares the predicted output with the ground truth label and returns whether
     * the prediction is correct.
     * 
     * @param output Predicted output vector.
     * @param label Ground truth label vector.
     * @return 1 if the prediction is correct, 0 otherwise.
     */
    int compare_result(const vector<double> &output, const vector<double> &label);

    /**
     * @brief Set the dropout probability for neurons in a specific layer.  
     * @param layer_num Index of the layer for which dropout probability is set.
     * @param probability Dropout probability.
     */
    void setDropout(unsigned int layer_num, double probability);

};
