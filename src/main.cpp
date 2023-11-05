#include <cassert>
#include <cmath>
#include <getopt.h>
#include <algorithm> // TODO can we use that?

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

void usage(){
    cerr << "Usage: ./network -e [NUM_EPOCHS] -l [LEARNING_RATE] -b [BATCH_SIZE] INPUT_NEURONS_AMOUNT HIDDEN_LAYER_1_NEURONS_AMOUNT [...] OUTPUT_NEURONS_AMOUNT" << endl;
}

int main(int argc, char *argv[]){
    unsigned epochs = 1;
    bool epochsSet = false;
    unsigned batchSize = 1;
    bool batchSizeSet = false;
    float learningRate = 0.01;
    bool learningRateSet = false;

    struct option long_options[] = {
        {"epochs", required_argument, nullptr, 'e'},
        {"learning_rate", required_argument, nullptr, 'l'},
        {"batch_size", required_argument, nullptr, 'b'},
        {nullptr, 0, nullptr, 0}
    };

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "e:l:b:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'e':
                epochs = std::atoi(optarg);
                epochsSet = true;
                break;
            case 'l':
                learningRate = std::atof(optarg);
                learningRateSet = true;
                break;
            case 'b':
                batchSize = std::atoi(optarg);
                batchSizeSet = true;
                break;
            case '?':
                std::cerr << "Unknown option or missing argument value" << std::endl;
                usage();
                return 1;
            default:
                std::cerr << "Unhandled option" << std::endl;
                usage();
                return 1;
        }
    }

    if(!epochsSet || !learningRateSet || !batchSizeSet){
        usage();
        return 1;
    }

    if (argc - optind < 3)
    {
        usage();
        return 1;
    }
    vector<unsigned> topology = parseTopology(argc - optind, &(argv[optind]));

    Neuron::setLearningRate(learningRate);

    Net myNet(topology);

    // TODO can we select the task using arguments, too? Or would that be too much?

    // InputData trainingInputs("./data/xor_inputs.csv", 1.0, batchSize);
    // LabelData trainingLabels("./data/xor_labels.csv", 1, true);
    // InputData testingInputs = trainingInputs;

    // InputData trainingInputs("./data/xor_inputs.csv", 1.0, batchSize);
    // LabelData trainingLabels("./data/and_or_labels.csv", 2, true);
    // InputData testingInputs = trainingInputs;

    // InputData trainingInputs("./data/iris_inputs.csv", 10.0, batchSize);
    // LabelData trainingLabels("./data/iris_labels.csv", 3, false);
    // InputData testingInputs = trainingInputs;

    InputData trainingInputs("./data/fashion_mnist_train_vectors.csv", 255.0, batchSize);
    LabelData trainingLabels("./data/fashion_mnist_train_labels.csv", 10, false);
    InputData testingInputs("./data/fashion_mnist_test_vectors.csv", 255.0, batchSize);

    vector<float> input, label, output;

    unsigned actual_batch_size;
    // TODO if batch size > size of dataset, batch size = size of dataset
    for(unsigned epoch = 0; epoch < epochs; ++epoch)
    {
        cout << "==================================================" << endl;
        cout << "Epoch " << epoch + 1 << endl;

        // Shuffle the training data
        unsigned seed = static_cast<unsigned>(time(nullptr));
        trainingInputs.shuffleData(seed);
        trainingLabels.shuffleData(seed);

        for(unsigned batch = 0; batch < ceil(trainingInputs.length() / batchSize); ++batch)
        {
            cout << "--------------------------------------------------" << endl;
            cout << "Batch " << batch + 1 << endl;
            actual_batch_size = trainingInputs.getNextBatchSize();
            myNet.resetGradientSum();

            for (unsigned i = 0; i < actual_batch_size; i++)
            {
                cout << "Epoch : Batch : Sample -> " << epoch + 1 << " : " << batch + 1 << " : " << i + 1 << endl;
                input = trainingInputs.getNext();
                label = trainingLabels.getNext();

                // showVectorVals(": Inputs:", input);
                myNet.feedForward(input);

                myNet.getResults(output);
                //cout << "out: " << m_layers.back()[batch].getOutputVal() << endl;
                //cout << "res val: " << output[0] << endl;
                showVectorVals("Outputs:", output);

                showVectorVals("Labels:", label);
                assert(label.size() == topology.back());
            
                myNet.backProp(label);

                cout << "Loss: " << myNet.getError() << endl;
                cout << "Avg loss: " << myNet.getRecentAverageError() << endl;
            }
            myNet.calcAvgGradient(actual_batch_size);
            myNet.updateWeights();
        }
    }

    cout << "Done training, begin testing" << endl;

    string test_labels_filepath = "test_labels.csv";
    ofstream file(test_labels_filepath);
    if(!file.is_open())
    {
        throw runtime_error("Unable to open file: " + test_labels_filepath);
    }
    for(unsigned i = 0; i < testingInputs.length(); ++i)
    {
        input = testingInputs.getNext();
        myNet.feedForward(input);
        myNet.getResults(output);

        // Get the index of the maximum value in the output vector
        auto maxElementIter = max_element(output.begin(), output.end());
        unsigned index = distance(output.begin(), maxElementIter);
        file << index << endl;
    }

    cout << "Done testing" << endl;
}
