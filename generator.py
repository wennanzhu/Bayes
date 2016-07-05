import datetime
import numpy as np

class Generator(object):
    def __init__(self):
        self.t = datetime.datetime(2016, 7, 04, 20, 0)

    def gendata(self, start, end, array1, array2, data_type):
        output = ""
        for i in range(start, end):
            self.t += datetime.timedelta(seconds = 1)
            output += self.t.strftime(':%Y-%m-%d %H%M%S%f')[:-3]+'-0500:' + str(array1[i]) + "," + str(array2[i]) + "," + str(data_type) + "\r\n"
        return output

def main():
    gen = Generator()

    a = np.ones(20000)
    b = np.append(np.random.normal(2, 0.1, 10000), np.random.normal(3, 0.1, 10000))
    outfile = open("data/bayes_training.txt", "w")
    output = "#a,b,type\r\n"
    output += gen.gendata(0, 10000, a, b, 1)
    output += gen.gendata(10000, 20000, a, b, 2)
    outfile.write(output)
    outfile.close()

    c = np.ones(30000) * 2
    d = np.append(np.random.normal(4, 0.1, 10000), np.random.normal(6, 0.1, 10000))
    d = np.append(d, np.random.normal(8, 0.1, 10000))

    outfile = open("data/bayes_testing.txt", "w")
    output = "#c,d,type\r\n"
    output += gen.gendata(0, 10000, c, d, 1)
    output += gen.gendata(10000, 20000, c, d, 2)
    output += gen.gendata(20000, 30000, c, d, 3)
    outfile.write(output)
    outfile.close()

if __name__ == "__main__":
    main()
