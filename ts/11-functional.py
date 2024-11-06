# Importing the necessary module
from functools import partial

# Define the list of numbers
x = list(range(1, 11))


# Define the function
def f(x):
    return x ** 2

# Apply the function to each element in x using map and convert to list
result = list(map(f, x))

# Print the result
print(result)

def g(x, y):
    z = x+y
    return z

a = ['a', 'b', 'c']
b = ['x', 'y', 'z']

#######################################


# Define the lists
x = ['a', 'b', 'c']
y = ['A', 'B', 'C']

# Define the function
def p(x, y):
    return f"{x}{y}"  # Concatenates without separator


p('c','d')

# Apply the function across pairs of x and y using map and zip
result = list(map(p, x, y))

# Print the result
print(result)
##############################################3
def f( i , x, y):
    return (x[i])**(y[i])

x = [1, 2, 5]
y = [1, 2, 5]
zip(x,y)
i = list(range(0,3))

list(?map(f , i, x, y ))
########################################################
from functools import partial
import itertools as it
# Define the lists
x = list(range(1, 6))
y = x  # y is the same as x
z = [0]

# Define the function
def h(x, y, z):
    return x + y - z
# non funziona
list(it.?imap(h, x, y, z))

