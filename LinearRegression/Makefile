CXX=g++
OBJECTS = LinearRegressionModel.o LinMain.o
EXEC=linreg
MOBJECTS = MLinearRegressionModel.o MLinMain.o ../Common/IRegressionModel.o ../Common/RangeNormalizer.o
MEXEC=mlinreg

all: $(EXEC) $(MEXEC)

$(EXEC): $(OBJECTS)
	       	$(CXX) -o $(EXEC) $(OBJECTS)


$(MEXEC): $(MOBJECTS)
	       	$(CXX) -o $(MEXEC) $(MOBJECTS)


clean:
	       	rm -f *.o $(EXEC) $(MEXEC)
