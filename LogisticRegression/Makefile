CXX=g++
CXXFLAGS = -std=c++11 -Wall
OBJECTS = LogisticRegressionModel.o LogMain.o ../Common/StDevNormalizer.o ../Common/MLUtils.o
EXEC=logreg
MOBJECTS = MLogisticRegressionModel.o LogisticRegressionModel.o MLogMain.o ../Common/StDevNormalizer.o ../Common/MLUtils.o
MEXEC=mlogreg

all: $(EXEC) $(MEXEC)

$(EXEC): $(OBJECTS)
	       	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJECTS)

$(MEXEC): $(MOBJECTS)
	       	$(CXX) $(CXXFLAGS) -o $(MEXEC) $(MOBJECTS)


clean:
	       	rm -f *.o $(EXEC) $(TESTS)

