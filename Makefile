SOURCES = 	src/main.cpp src/main.hpp \
		src/input_data.cpp src/input_data.hpp \
		src/label_data.cpp src/label_data.hpp \
		src/neuron.cpp src/neuron.hpp \
		src/net.cpp src/net.hpp


all: network


network: $(SOURCES)
	g++ -std=c++17 -Wall -O3 -Ofast $(SOURCES) -o network -g


run: network
	./network 2 2 1


clean:
	rm network
