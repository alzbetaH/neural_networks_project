SOURCES = 	src/main.cpp src/main.hpp \
			src/training_data.cpp src/training_data.hpp \
			src/neuron.cpp src/neuron.hpp \
			src/net.cpp src/net.hpp

all: network

# dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
network: $(SOURCES)
	@echo TODO COMPILE
	# g++ -std=c++17 -Wall -O3 -Ofast src/... -o network
	g++ -std=c++17 -Wall -O3 -Ofast $(SOURCES) -o network 


run: network
	@echo TODO RUN


clean:
	@echo TODO CLEAN
	rm network
