CXX=g++
CXXFLAGS = -std=c++11 -Wall
OBJECTS = LinearRegressionModel.o LinMain.o
EXEC=linreg
MOBJECTS = MLinearRegressionModel.o MLinMain.o ../Common/RangeNormalizer.o
MEXEC=mlinreg

all: $(EXEC) $(MEXEC)

$(EXEC): $(OBJECTS)
	       	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJECTS)


$(MEXEC): $(MOBJECTS)
	       	$(CXX) $(CXXFLAGS) -o $(MEXEC) $(MOBJECTS)


clean:
	       	rm -f *.o $(EXEC) $(MEXEC)
