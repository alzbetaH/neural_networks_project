SOURCES = 	src/main.cpp src/main.hpp \
		src/input_data.cpp src/input_data.hpp \
		src/label_data.cpp src/label_data.hpp \
		src/neuron.cpp src/neuron.hpp \
		src/net.cpp src/net.hpp


all: network


network: $(SOURCES)
	g++ -std=c++17 -Wall -O3 -Ofast $(SOURCES) -o network -g


run: mnist
	./network -e 7 -b 32 -l 0.001 784 64 32 10


pack: clean
	./pack.sh


clean:
	rm -f network, train_predictions.csv test_predictions.csv xhrabos_xskalos.zip
