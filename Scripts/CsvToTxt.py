print 'Input path:'
inputPath = raw_input()
print 'Output path:'
outputPath = raw_input()

def csvToTxt(inputPath, outputPath):
    i = open(inputPath)
    o = open(outputPath, 'w')

    o.write(i.read().replace(',', ' '))

csvToTxt(inputPath, outputPath)
