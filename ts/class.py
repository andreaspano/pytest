class persona:

    def __init__(self, h):
        self.h = h

    def w(self):
        w = self.h -100
        return(w)


andrea = persona(h = 190)
andrea.h
andrea.w()


class persona:




class MyClass:

    @classmethod
    def from_alternativeConstructor(cls):      # Alternative Constructor
        return cls()                           # returns object

object = MyClass.from_alternativeConstructor() 


#####################################

from dataclasses import dataclass

@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
##################################################3
@dataclass
class persona():
    h: int = 100

    def w(self):
        w = self.h - 100 
        return w

io = persona(190)

io.h
io.w()
#####################################

# main.py
from dataclasses import dataclass
from math import sqrt, pow

@dataclass
class Point:
    x: int = 0
    y: int = 0

    def distance_to(self, p) -> float:
        return sqrt(pow(p.x - self.x, 2) + pow(p.y - self.y, 2))

p1 = Point()
p2 = Point(10, 20)

# prints 22.360679774997898
print(p1.distance_to(p2))
