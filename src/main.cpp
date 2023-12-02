#include <cassert>
#include <cmath>
#include <getopt.h>
#include <algorithm> // TODO can we use that?
#include <cfenv>

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

void showVectorVals(string label, vector<double> &v)
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
    // feenableexcept(FE_ALL_EXCEPT);
    unsigned epochs = 1;
    bool epochsSet = false;
    unsigned batchSize = 1;
    bool batchSizeSet = false;
    double learningRate = 0.01;
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

    // TODO Find a good seed and set it
    unsigned seed = static_cast<unsigned>(time(nullptr));
    // unsigned seed = 42;

    Net myNet(topology, seed);

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
    LabelData testingLabels("./data/fashion_mnist_test_labels.csv", 10, false);

    // split into training and validation data
    trainingInputs.splitData(0.8);
    trainingLabels.splitData(0.8);

    // string accuracy_file_path = "accuracy.csv";
    // ofstream file_acc(accuracy_file_path);
    // if(!file_acc.is_open())
    // {
    //     throw runtime_error("Unable to open file: " + accuracy_file_path);
    // }

    vector<double> input, label, output;
    vector<double> input_v, label_v, output_v;
    vector<double> input_t, label_t, output_t;

    unsigned actual_batch_size;
    // TODO if batch size > size of dataset, batch size = size of dataset
    for(unsigned epoch = 0; epoch < epochs; ++epoch)
    {
        cout << "==================================================" << endl;
        cout << "Epoch " << epoch + 1 << endl;

        // Shuffle the training data
        trainingInputs.shuffleData(seed);
        trainingLabels.shuffleData(seed);

        // set dropout (hidden layers only)
        myNet.setDropout(1, 0.5);
        myNet.setDropout(2, 0.05);

        for(unsigned batch = 0; batch < ceil(trainingInputs.trainLength() / batchSize); ++batch)
        {
            // cout << "--------------------------------------------------" << endl;
            // cout << "Batch " << batch + 1 << endl;
            actual_batch_size = trainingInputs.getNextBatchSize();
            myNet.resetGradientSum();

            for (unsigned i = 0; i < actual_batch_size; i++)
            {
                // cout << "Epoch : Batch : Sample -> " << epoch + 1 << " : " << batch + 1 << " : " << i + 1 << endl;
                input = trainingInputs.getNextTrain();
                label = trainingLabels.getNextTrain();

                // showVectorVals(": Inputs:", input);
                myNet.feedForward(input);

                myNet.getResults(output);
                //cout << "out: " << m_layers.back()[batch].getOutputVal() << endl;
                //cout << "res val: " << output[0] << endl;
                // showVectorVals("Outputs:", output);

                // showVectorVals("Labels:", label);
                // assert(label.size() == topology.back());
            
                myNet.backProp(label);

                // cout << "Loss: " << myNet.getError() << endl;
                // cout << "Avg loss: " << myNet.getRecentAverageError() << endl;
            }
            myNet.calcAvgGradient(actual_batch_size);
            myNet.updateWeights();
        }

        // counting validation loss and accuracy
        double accuracy_sum_v = 0;
        double avg_loss_v = 0;

        // Unset dropout for all layers
        for(unsigned layerNum = 0; layerNum < topology.size(); ++layerNum)
        {
            myNet.setDropout(layerNum, 0.0);
        }

        for(unsigned j = 0; j < trainingInputs.validLength(); ++j)
        {
            input_v = trainingInputs.getNextValid();
            label_v = trainingLabels.getNextValid();
            myNet.feedForward(input_v);
            myNet.getResults(output_v);
            if (myNet.compare_result(output_v, label_v))
            {
                accuracy_sum_v++;
            }
            avg_loss_v += myNet.getLoss(label_v);
        }
        showVectorVals("Outputs:", output_v);
        cout << "Validation Loss: " << avg_loss_v / trainingInputs.validLength() << endl;
        cout << "Validation Accuracy: " << accuracy_sum_v / trainingInputs.validLength() << endl;
        //file_acc << "validation;" << epoch + 1 << ";" <<  accuracy_sum_v / trainingInputs.validLength() << endl;
        
        // counting train loss and accuracy
        double accuracy_sum_t = 0;
        double avg_loss_t = 0;
        
        for(unsigned j = 0; j < trainingInputs.trainLength(); ++j)
        {
            input_t = trainingInputs.getNextTrain();
            label_t = trainingLabels.getNextTrain();
            myNet.feedForward(input_t);
            myNet.getResults(output_t);
            if (myNet.compare_result(output_t, label_t))
            {
                accuracy_sum_t++;
            }
            avg_loss_t += myNet.getLoss(label_t);
        }
        cout << "Train Loss: " << avg_loss_t / trainingInputs.trainLength() << endl;
        cout << "Train Accuracy: " << accuracy_sum_t / trainingInputs.trainLength() << endl;
        //file_acc << "train;" << epoch + 1 << ";" <<  accuracy_sum_t / trainingInputs.trainLength() << endl;
    }
    cout << "----------------------------------" << endl;
    cout << "Done training, begin testing" << endl;

    // string test_labels_filepath = "test_labels.csv";
    // ofstream file(test_labels_filepath);
    // if(!file.is_open())
    // {
    //     throw runtime_error("Unable to open file: " + test_labels_filepath);
    // }

    double accuracy_sum_test = 0;
    for(unsigned i = 0; i < testingInputs.length(); ++i)
    {
        input = testingInputs.getNext();
        label = testingLabels.getNext();
        myNet.feedForward(input);
        myNet.getResults(output);

        // Get the index of the maximum value in the output vector
        auto maxElementIter = max_element(output.begin(), output.end());
        unsigned index = distance(output.begin(), maxElementIter);

        if (myNet.compare_result(output, label))
        {
            accuracy_sum_test++;
        }
        
        //file << index << endl;
    }
    cout << "accuracy_test:" << accuracy_sum_test / testingLabels.length() << endl;
    cout << "Done testing" << endl;
}
