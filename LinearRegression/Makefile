CXX_FLAGS = -std=c++11
CXX=g++
OBJECTS = LinearRegressionModel.o main.o
EXEC=linreg

$(EXEC): $(OBJECTS)
	       	$(CXX) $(CXX_FLAGS) -o $(EXEC) $(OBJECTS)


clean:
	       	rm -f *.o $(EXEC) $(TESTS)

