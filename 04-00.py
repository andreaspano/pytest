class Person:
    "A simple class def"
    def __init__(self, name , age):
        self.name = name
        self.age = age
    
io = Person("Andrea", 55)    
io.name
io.age
print(dir(Person))
print(dir(io))

print(io.__doc__)
print(Person.__name__)
print(Person.__doc__)
print(io.__class__)
print(io.__dict__)


class Person:
    "A simple class def"
    def __init__(self, name , age):
        self.name = name
        self.age = age
    def y(self, year):
        year = year-self.age
        return year
    
        

io = Person("Andrea", 55)    
print(io.y(2024))



class Car:
    "a car class definition"
    def __init__(self, kml, disp):
        self.kml = kml
        self.disp = disp
    def mpg(self, n):
        mpg = self.kml/n
        return mpg
    
     
        
        
        


freemont = Car(kml = 10, disp = 3000)
freemont.mpg(n = 4)

freemont.disp


# ---------------  Class Robot -------------------#
class Robot:
    id = 1
    def __init__ (self, owner):
        self.owner = owner
        self.id = self.next_id()
    def next_id(self):
        id = self.id+1
        return id
        
    
R1 = Robot(owner = 'Andrea') 
R1.owner

# ---------------  Class Employee -------------------#

class Employee:
    def __init__(self, name, position):
        self.name = name
        self.position = position
    
    promotion_table = {
        'A' : 'B',
        'B' : 'C',
        'C' : 'D'
        
    }
    
    def promote(self):
        self.position = Employee.promotion_table[self.position]
        
    def print(self):
        print( self.name, "%s" % self.position)
        
    


emp = Employee('Andrea', 'B')
Employee.print(emp)
emp.promote()



class Employee:
    def __init__(self, name, position):
        self.name = name
        self.position = position
    
    promotion_table = {
        'A' : 'B',
        'B' : 'C',
        'C' : 'D'
        }
    
    def promote(self):
        self.position = Employee.promotion_table[self.position]
        
    def print(self):
        print( self.name, "%s" % self.position, end = ' ')
        
    


emp = Employee('Andrea', 'B')
Employee.print(emp)
emp.promote()
Employee.print(emp)


class Manager(Employee ):
    def __init__(self, name, position , dep):
        Employee.__init__(self, name, position)
        self.dep = dep
    def print(self):
        Employee.print(self)
        print('dep:', self.dep)

man = Manager('Ugo', 'B', 'ortofrutta' )
Manager.print(man)

class Manager(Employee ):
    def __init__(self, name, position , dep):
        Employee.__init__(self, name, position)
        self.dep = dep
        
    def promote(self):
        if self.position == 'D':
            print('no more')
            return
    def print(self):
        super(Manager, self).print()
        print('dep:', self.dep)
        

man = Manager('Ugo', 'D', 'ortofrutta' )
man.promote()

Manager.print(man)
print(Manager.man)




class Manager(Employee ):
    def __init__(self, name, position , dep):
        super(Manager, self).__init__(name, position)
        self.dep = dep
        
    def promote(self):
        if self.position == 'D':
            print('no more')
            return
    def print(self):
        super(Manager, self).print()
        print('dep:', self.dep)
        

man = Manager('Ugo', 'D', 'ortofrutta' )
man.promote()

Manager.print(man)

# Diamond ineritahnace

class A():
    def __init__(self): 
        print('class A init')

class B(A):
    def __init__(self): 
        print('class B init')
        super().__init__()
        
class C(A):
    def __init__(self): 
        print('class C init')
        super().__init__()

class D(B, C):
    def __init__(self): 
        print('class D init')
        super().__init__()



d = D()

# ----------------- iter

class X:
    def __init__(self, arr):
       self.arr = arr
    def __call__(self):
        return self.arr

x = X([1,2,3])
x

class X:
    def __init__( self, arr ) :
        self.arr = arr
        
    def __call__(self):
        return self.arr


X([1,2])

class Y:
    def __init__(self, arr):
        self.arr = arr
        
    def __call__(self):
        print ("invoking __call__ method")
    
    def __iter__(self):
        return Yiter(self)
                    
class Yiter:
    def __init__(self, y):
        self.item = y.arr
        self.index = -1
    def __iter__(self):
        return self
    def __next__(self):
        self.index += 1
        if self.index < len(self.item):
            return self.item[self.index]
        else:
            print ('tail')
            raise StopIteration
    
    next = __next__
        
    dir(y)
    



y = Y([1, 2,3, 4, 5])
y.arr
y()
it = iter(y)
print(it.next())

        
    




