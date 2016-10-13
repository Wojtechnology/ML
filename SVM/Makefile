CXX=g++
CXXFLAGS = -std=c++11 -Wall
OBJECTS = SVMModel.o SVMMain.o SVMKernel.o ../Common/StDevNormalizer.o ../Common/MLUtils.o
EXEC=svm

all: $(EXEC) $(MEXEC)

$(EXEC): $(OBJECTS)
	       	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJECTS)


clean:
	       	rm -f *.o $(EXEC) $(TESTS)

