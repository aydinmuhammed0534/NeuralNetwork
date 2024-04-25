import numpy as np
import random as rn          

def sigmoid(x):                                
 return 1 / (1 + np.exp(-x))                       

class Neuron:
    def __init__(self,weights,bias) :
        self.weights=weights
        self.bias=bias
          
    def feedforward(self, inputs):                
        output = np.dot(inputs,self.weights) + self.bias 
        return sigmoid(output) 
 
data = {
     "Green":1,"Black":2 ,"Blue":3 ,"Brown":4 ,"Hazel":5 ,
     "Blonde":6 ,"Pink":7 ,"Purple":8    
}

x=np.array([[175,68,data["Black"],data["Blue"]],[150,52,data["Blonde"],data["Hazel"]]])
bias = np.random.uniform(-1, 1,size=2)

weights = np.random.uniform(-1,1,size=(4,128))
n = Neuron(weights,bias[0])
p1=n.feedforward(x)

weights = np.random.uniform(-1,1,size=(128,1))
n2=Neuron(weights,bias[1])
p2=n2.feedforward(p1)
print(p2)


#araya katman ekleme denemesi 
# weights = np.random.uniform(-1,1,size=(128,256))
# n2= Neuron(weights,bias[1])
# p2=n2.feedforward(p1)
    
# weights = np.random.uniform(-1,1,size=(256,1))
# n3= Neuron(weights,bias[2])
# p3=n3.feedforward(p2)
