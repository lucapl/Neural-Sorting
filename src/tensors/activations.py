from tensorflow.keras import backend as K

delta = 0.1
def leaky_hard_sigmoid(x):
    return K.switch(x < 0, 
                    x*delta, 
                    K.switch(
                        x > 1,
                        delta*(x-1) + 1,
                        x
                    ))