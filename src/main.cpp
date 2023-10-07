#include <cassert>
#include <cmath>

#include "main.hpp"

vector<unsigned int> parseTopology(int number_of_layers, char* neurons_per_layer[]) {
    /**
     * @brief Parse strings in `neurons_per_layer` as the number of neurons in
     * the network layers, describing the whole network topology.
     * 
     * Raises invalid_argument if unsuccessful
     */
    vector<unsigned int> numbers;

    for (int i = 0; i < number_of_layers; ++i) {
        string arg = neurons_per_layer[i];

        // Convert string to int and check if the whole string was processed
        size_t pos;
        unsigned int num = stoul(arg, &pos);
        if (pos != arg.size()) {
            throw invalid_argument("Not a valid unsigned integer: " + arg);
        }
        numbers.push_back(num);
    }

    return numbers;
}

void showVectorVals(string label, vector<float> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}

int main(int argc, char *argv[]){
    if (argc < 3)
    {
        cerr << "Usage: ./network INPUT_NEURONS_AMOUNT HIDDEN_LAYER_1_NEURONS_AMOUNT [...] OUTPUT_NEURONS_AMOUNT" << endl;
        exit(1);
    }

    // We can still fit arguments for the learning rate, batch size and such,
    // here, I guess

    unsigned epochs = 2000; // TODO Make it an argument
    // TODO Learning rate
    // TODO Batch size?

    vector<unsigned> topology = parseTopology(argc - 1, &(argv[1]));

    Net myNet(topology);

    // InputData trainingInputs("./data/fashion_mnist_train_vectors.csv", 255.0);
    // LabelData trainingLabels("./data/fashion_mnist_train_labels.csv", 10, false);
    // InputData testingInputs("./data/fashion_mnist_test_vectors.csv", 255.0);

    InputData trainingInputs("./data/xor_inputs.csv", 1.0);
    LabelData trainingLabels("./data/xor_labels.csv", 1, true);

    vector<float> input, label, output;

    for(unsigned epoch = 0; epoch < epochs; ++epoch)
    {
        cout << "==================================================" << endl;
        cout << "Epoch " << epoch + 1 << endl;

        // Shuffle the training data
        unsigned seed = static_cast<unsigned>(time(nullptr));
        trainingInputs.shuffleData(seed);
        trainingLabels.shuffleData(seed);

        for(unsigned i = 0; i < trainingInputs.length(); ++i)
        {
            cout << "Epoch " << epoch + 1 << ", pass " << i + 1 << endl;

            input = trainingInputs.getNext();
            label = trainingLabels.getNext();

            // showVectorVals(": Inputs:", input);
            myNet.feedForward(input);

            myNet.getResults(output);
            //cout << "out: " << m_layers.back()[i].getOutputVal() << endl;
            //cout << "res val: " << output[0] << endl;
            showVectorVals("Outputs:", output);

            showVectorVals("Targets:", label);
            assert(label.size() == topology.back());
        
            myNet.backProp(label);

            cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
        }
    }

    cout << endl << "Done" << endl;
}
