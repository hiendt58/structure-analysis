import numpy as np 
from io import StringIO

def load_structure_data(num_bars=10, num_exams=500, ele="", scale=0.2):
    # ele: number of outputs
    if ele == "":
        inp_file = "data/truss_%d_%d.txt" %(num_bars, num_exams)
    else:
        inp_file = "data/truss_%d_%d_%s.txt" %(num_bars, num_exams, ele)
    data = load_from_file(inp_file)
    X = data[:,:num_bars]
    y = data[:,num_bars:]
    
    # normalize input with 35.0
    X = X / 35.0
    # normalize range for output
    # lower = -2.0
    # upper = 2.0
    if ele == "18":
        y[:8] = y[:8] / np.max(abs(y[:8]))
        y[8:] = y[8:] / np.max(abs(y[8:]))
    else:
        y = y / np.max(abs(y))
    return test_train_split(X,y,scale)

def load_from_file(filename):
    f = open(filename, "r")
    string_data = f.read()
    string = StringIO(string_data)
    data = np.loadtxt(string)
    return data

# def in_out_split()
    
def test_train_split(X, y, scale):
    if X.shape[0] != y.shape[0]:
        print("Input, output are not the same size!")
        return
    size = X.shape[0]
    test_size = int(size * scale)
    X_test = X[:test_size]
    y_test = y[:test_size]
    X_train = X[test_size:]
    y_train = y[test_size:]
    return X_train, y_train, X_test, y_test

def normalize(data, lower, upper):
    max_val = np.max(data)
    min_val = np.min(data)
    data_norm = np.array([(i - min_val)*(upper-lower)/(max_val-min_val) + lower for i in data])
    return data_norm