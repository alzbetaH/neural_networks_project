/**
 * @file neuron.hpp
 * @brief Declaration of the Neuron class and its methods.
 */

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>

using namespace std;

/**
 * @class Neuron
 * @brief Represents a single neuron in a neural network.
 */
class Neuron;

/**
 * @typedef Layer
 * @brief Represents a layer of neurons in a neural network.
 *
 * A Layer is defined as a vector of Neuron objects. It is used to organize neurons
 * within a neural network and facilitate the flow of information between layers.
 */
typedef vector<Neuron> Layer;

/**
 * @struct Connection
 * @brief Represents a connection between neurons in a neural network.
 *
 * The Connection struct holds information about the weight of a connection between two neurons
 * and the change in weight (delta weight) during training. It is used to store connection
 * properties and update weights in the neural network.
 */
struct Connection
{
    double weight;
    double deltaweight;
};

class Neuron
{
private:
    /**
     * @brief Overall learning rate for the neuron.
     */
    static double eta;

    /**
     * @brief Momentum multiplier for weight updates.
     */
    static double alpha;
    
    /**
     * @brief Decay factor for RMSprop optimization.
     */
    static double decay;

    /**
     * @brief Small constant used to prevent division by zero.
     */
    static double epsilon;

    /**
     * @brief He weight initialization for the neuron.
     * @param numLayerInputs Number of inputs from the previous layer.
     * @return Initialized weight value.
     */
    static double heWeightInit(unsigned numLayerInputs, unsigned seed);

    /**
     * @brief Calculate the sum of derivatives of weights for the neuron.
     * @param nextLayer The next layer of neurons.
     * @return The sum of derivatives of weights.
     */
    double sumDOW(const Layer &nextLayer) const;

    /**
     * @brief Apply the ReLU transfer function to the given value.
     * @param x Input value.
     * @return Transformed output value.
     */
    static double transferFunction(double x);

    /**
     * @brief Calculate the derivative of the ReLU transfer function.
     * @param x Input value.
     * @return Derivative value.
     */
    static double transferFunctionDerivative(double x);

    /**
     * @brief Inner potential of the neuron.
     */
    double m_potential;

    /**
     * @brief Output value of the neuron after using the activation function.
     */
    double m_outVal; 

    /**
     * @brief Output weights for connections to neurons in the next layer.
     */ 
    vector<double> m_outWeights;

    /**
     * @brief Previous weight changes for momentum.
     */
    vector<double> m_outWeightsDeltas;

    /**
     * @brief Gradients of the output weights.
     */
    vector<double> m_outWeightsGradients;

    /**
     * @brief The neuron index (m_myIndex) indicates the position of the neuron within its layer.
     */
    unsigned m_myIndex;

    /**
     * @brief Gradient of the neuron representing the rate of change of the loss with respect to
     * the inner potential of the neuron.
     */
    double m_gradient;

    /**
     * @brief Probability of neuron being dropped out.
     */
    double dropout_probability; // Probability of neuron being dropped out

    /**
     * @brief Probability of neuron being dropped out, but in interval from 0 to RAND_MAX.
     */
    int dropout_probability_int; 

public:
    /**
     * @brief Constructor for the Neuron class.
     * @param numLayerInputs Number of inputs from the previous layer.
     * @param numNeuronOutputs Number of outputs from the neuron.
     * @param neuronIndex Index of the neuron in the layer.
     */
    Neuron(unsigned numLayerInputs, unsigned numNeuronOutputs, unsigned neuronIndex, unsigned seed);

    /**
     * @brief Set the learning rate for the neuron.
     * @param learningRate Learning rate value.
     */
    static void setLearningRate(double learningRate);

    /**
     * @brief Set the dropout probability for the neuron.
     * @param probability Dropout probability.
     */
    void setDropout(double probability);

    /**
     * @brief Set the output value of the neuron.
     * @param val The new output value to be set.
     */
    void setOutputVal(double val) { m_outVal = val; }

    /**
     * @brief Get the inner potential of the neuron.
     * @return The inner potential of the neuron.
     */
    double getPotential() { return m_potential; }

    /**
     * @brief Get the current output value of the neuron.
     * @return The current output value of the neuron.
     */
    double getOutputVal(void) const { return m_outVal; }

    /**
     * @brief Update the weights of the neuron using the RMSprop optimization algorithm.
     */
    void updateWeights();

    /**
     * @brief Calculate the gradients for hidden layer neurons.
     * @param nextLayer The next layer of neurons.
     */
    void calcHiddenGradients(const Layer &nextLayer);
    
    /**
     * @brief Calculate the gradients for output layer neurons (difference between output value and desired value).
     * @param targetVal Desired value for the neuron.
     */
    void calcOutputGradients(double targetVal);

    /**
     * @brief Calculate the weight gradients for the neuron.
     * @param nextLayer The next layer of neurons.
     */
    void calcWeightGradients(const Layer &nextLayer);
    
    /**
     * @brief Calculate the inner potential of the neuron based on the previous layer's output.
     * @param prevLayer The previous layer of neurons.
     */
    void calcPotential(const Layer &prevLayer);

    /**
     * @brief Apply the transfer function to calculate the output value of the neuron.
     */
    void calcOutput();

    /**
     * @brief Calculate the average batch gradient for the neuron's weights.
     * @param batchSize Size of the training batch.
     */
    void calcAvgGradient(unsigned int batchSize);

    /**
     * @brief Reset the accumulated gradients to zero.
     */
    void resetGradientSum();

};