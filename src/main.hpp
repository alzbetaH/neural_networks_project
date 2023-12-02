/**
 * @file main.hpp
 * @brief Declaration of the main functions for training and testing the neural network.
 */

#include <getopt.h>
#include <iostream>
#include "net.hpp"
#include "input_data.hpp"
#include "label_data.hpp"

/**
 * @brief Parse strings in `neurons_per_layer` as the number of neurons in the network layers,
 * describing the whole network topology.
 * 
 * @param number_of_layers Number of layers in the network.
 * @param neurons_per_layer An array representing the number of neurons in each layer.
 * @return A vector of unsigned integers representing the network topology.
 * @throws std::invalid_argument if unsuccessful.
 */
vector<unsigned int> parseTopology(int number_of_layers, char* neurons_per_layer[]);

/**
 * @brief Display the correct syntax by running program.
 */
void usage();

/**
 * @brief Test the network and save predictions to a file.
 * 
 * @param myNet The neural network.
 * @param inputs The input data for testing.
 * @param output_filepath The filepath to save the predictions.
 */
void testAndSavePredictions(Net &myNet, InputData &inputs, string output_filepath);

/**
 * @brief Test the network on a dataset and print accuracy.
 * 
 * @param myNet The neural network.
 * @param inputs The input data for testing.
 * @param labels The labels for testing.
 * @param subsetName The name of the dataset subset (e.g., "Training", "Testing").
 */
void testAndPrintAccuracy(Net &myNet, InputData &inputs, LabelData &labels, string subsetName);

/**
 * @brief The main function for training and testing the neural network.
 * 
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Exit status.
 */
int main(int argc, char *argv[]);