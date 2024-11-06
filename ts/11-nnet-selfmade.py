import numpy as np
from dataclasses import dataclass


# def sigmoid function 
def sigmoid(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  deriv =  fx * (1 - fx)
  return deriv


# def neuron class
@dataclass
class Neuron:
    weights: float
    bias: float

   
    def feedforward(self, inputs) -> float:
    # Weight inputs, add bias, then use the activation function
        #total = np.dot(self.weights , inputs) + self.bias
        total = np.sum(self.weights * inputs) + self.bias
        return sigmoid(total)

#test
x = np.array([2, 3])
weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)
n.feedforward(x)
##################################################################
@dataclass
class nnet1:
    weights: float = np.array([0,1])
    bias: float = 0 
    
    def __post_init__(self):
        self.h1 = Neuron(self.weights, self.bias)
        self.h2 = Neuron(self.weights, self.bias)
        self.o1 = Neuron(self.weights, self.bias)
    
    
    def run_net(self, x):
        
        #hidden layer
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_h = np.array([out_h1, out_h2])

        out_o1 = self.o1.feedforward(out_h)

        return out_o1


# test
model = nnet1()
model.run_net(x) # 0.7216325609518421
###############################################################
###############################################################
###############################################################
###############################################################
# def sigmoid function 
def sigmoid(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  deriv =  fx * (1 - fx)
  return deriv

# def mse
def mse_loss(y, prd):
  mse =  ((y - prd) ** 2).mean()
  return mse



def feedforward(x, weights, bias) -> float:
  x = np.array(x)
  weights = np.array(weights)
  bias = np.array(bias)
  total = np.sum(weights * x) + bias
  return total, sigmoid(total)


# data
X = np.array([
  [-200, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

Y = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

i = 1

x = X[i]
y = Y[i]
#x = [-2, 1]
#y = 1
lr = 0.01
loss = np.array([], dtype = float)




#Weights
w1 = np.random.uniform(size = 1)
w2 = np.random.uniform(size = 1)        
w3 = np.random.uniform(size = 1)
w4 = np.random.uniform(size = 1)
w5 = np.random.uniform(size = 1)        
w6 = np.random.uniform(size = 1)


# Biases
b1 = np.random.normal(size = 1)
b2 = np.random.normal(size = 1)
b3 = np.random.normal(size = 1)

# Run nnet

h1, h1s = feedforward(x , [w1,w2], b1)
h2, h2s = feedforward(x , [w3, w4], b2)
oo, oos = feedforward([h1, h2] , [w5,w6], b3) 

prd = oos

#------ backward ------#
L_prd = -2 * (y - prd)

# Neuron oo
prd_w5 = h1s * deriv_sigmoid(oo)
prd_w6 = h2s * deriv_sigmoid(oo)
prd_b3 = deriv_sigmoid(oo)
prd_h1 = w5 * deriv_sigmoid(oo)
prd_h2 = w6 * deriv_sigmoid(oo)

# Neuron h1
h1_w1 = x[0] * deriv_sigmoid(h1)
h1_w2 = x[1] * deriv_sigmoid(h1)
h1_b1 = deriv_sigmoid(h1)

# Neuron h2
h2_w3 = x[0] * deriv_sigmoid(h2)
h2_w4 = x[1] * deriv_sigmoid(h2)
h2_b2 = deriv_sigmoid(h2)

#------ Update ------#

# Neuron h1
w1 -= lr * L_prd * prd_h1 * h1_w1
w2 -= lr * L_prd * prd_h1 * h1_w2
b1 -= lr * L_prd * prd_h1 * h1_b1

# Neuron h2
w3 -= lr * L_prd * prd_h2 * h2_w3
w4 -= lr * L_prd * prd_h2 * h2_w4
b2 -= lr * L_prd * prd_h2 * h2_b2

# Neuron oo
w5 -= lr * L_prd * prd_w5
w6 -= lr * L_prd * prd_w6
b3 -= lr * L_prd * prd_b3



# --- Calculate total loss at the end of each epoch
#      if epoch % 10 == 0:
#y_preds = np.apply_along_axis(self.feedforward, 1, data)
#y_preds = np.apply_along_axis(feedforward, 1, x)
l = mse_loss(y, prd)

loss = []
loss.append(l)
print(loss)

h = np.array([])
k = np.array([.11, 7.432, 3])
np.concatenate((h,k))


import pandas as pd
df = pd.DataFrame({'x' : [1,2,3], 'y' : [3, 1, 2]})

def g(x):
  return x[0] + x[1]


x = [1,2, 3] 
y = [3,4, 5]
i = 1
for  x,y in zip(x,y):
  print(x)

  
  
   )
list(zip(x,y))

g(df)


np.apply_along_axis(g, 1, df)

df[:1]

###########################

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])


for x, y_true in zip(data, all_y_trues):
  sum_h1 = w1 * x[0] + w2 * x[1] + b1
  h1 = sigmoid(sum_h1)
