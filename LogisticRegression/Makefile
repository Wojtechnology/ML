CXX=g++
CXXFLAGS = -std=c++11 -Wall
OBJECTS = LogisticRegressionModel.o LogMain.o ../Common/RangeNormalizer.o
EXEC=logreg

$(EXEC): $(OBJECTS)
	       	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJECTS)


clean:
	       	rm -f *.o $(EXEC) $(TESTS)

