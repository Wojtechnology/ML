CXX=g++
CXXFLAGS = -std=c++11 -Wall
OBJECTS = NeuralNetworkModel.o NNMain.o ../Common/MLUtils.o ../Common/StDevNormalizer.o
EXEC=nn

all: $(EXEC) $(MEXEC)

$(EXEC): $(OBJECTS)
	       	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJECTS)


clean:
	       	rm -f *.o $(EXEC) $(TESTS)

