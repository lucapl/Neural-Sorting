import numpy as np

from collections import Counter

def undersample(X,y,samples=None,class_0=0):
    counts = Counter(y)
    samples = min(counts.values()) if samples == None else samples
    n = len(X)

    new_X = []
    new_y = []
    for i in range(class_0,len(counts)+class_0):
        ids = np.arange(len(y))
        current_class = ids[y==i]

        sampled = np.random.choice(current_class,samples)
        new_X.append(X[sampled])
        new_y.append(y[sampled])

    new_data = np.column_stack((np.concatenate(new_X,axis=0),np.concatenate(new_y,axis=0)))

    np.random.shuffle(new_data)

    return new_data[:,:-1],new_data[:,-1]