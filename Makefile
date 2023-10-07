SOURCES = 	src/main.cpp src/main.hpp \
		src/input_data.cpp src/input_data.hpp \
		src/label_data.cpp src/label_data.hpp \
		src/neuron.cpp src/neuron.hpp \
		src/net.cpp src/net.hpp


all: network


network: $(SOURCES)
	g++ -std=c++17 -Wall -O3 -Ofast $(SOURCES) -o network -g


xor:
	./network -e 2000 -b 1 -l 0.1 2 2 1


mnist:
	./network -e 5 -b 1 -l 0.01 784 256 64 10


run: network xor
	@


clean:
	rm network
