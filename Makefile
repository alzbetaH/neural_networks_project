SOURCES = 	src/main.cpp src/main.hpp \
		src/input_data.cpp src/input_data.hpp \
		src/label_data.cpp src/label_data.hpp \
		src/neuron.cpp src/neuron.hpp \
		src/net.cpp src/net.hpp


all: network


network: $(SOURCES)
	g++ -std=c++17 -Wall -O3 -Ofast $(SOURCES) -o network -g


xor: network
	./network -e 2000 -b 1 -l 0.1 2 2 1
	python3 evaluator/evaluate.py ./test_labels.csv ./data/xor_labels.csv


iris: network
	./network -e 1000 -b 1 -l 0.01 4 10 3
	python3 evaluator/evaluate.py ./test_labels.csv ./data/iris_labels.csv


mnist: network
	./network -e 35 -b 32 -l 0.001 784 64 16 10


run: mnist
	@


pack: clean
	./pack.sh


clean:
	rm -f network, train_predictions.csv test_predictions.csv xhrabos_xskalos.zip
