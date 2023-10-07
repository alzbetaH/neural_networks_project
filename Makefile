

all: network

# dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
network: src/main.cpp
	@echo TODO COMPILE
	# g++ -std=c++17 -Wall -O3 -Ofast src/... -o network
	g++ -std=c++17 -Wall -O3 -Ofast src/main.cpp -o network 


run: network
	@echo TODO RUN


clean:
	@echo TODO CLEAN
	rm network
