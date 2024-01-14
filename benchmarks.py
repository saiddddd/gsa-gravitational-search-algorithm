import numpy as np 
import math
import random


def prod( it ):
    p= 1
    for n in it:
        p *= n
    return p

def Ufun(x,a,k,m):
    y=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a));
    return y


def F1(x):
    """ Spere Function """
    s=np.sum(x**2);
    return s

def F2(x):    
    return sum(abs(xi) for xi in x) + prod(abs(xi) for xi in x)

def F3(x):
    dimension = len(x)
    R = 0
    for i in range(dimension):
        R += sum(x[:i+1])**2
    return R

def F4(x):
        return max(abs(xi) for xi in x)

def F5(x):
    dimension = len(x)
    return sum(100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(dimension-1))

def F6(x):
    return sum(int(xi + 0.5)**2 for xi in x)

def F7(x):
    dimension = len(x)
    return sum([(i+1)*(xi**4) for i, xi in enumerate(x)]) + random.random()

def F8(x):
    return sum(-xi * math.sin(math.sqrt(abs(xi))) for xi in x)

def F9(x):
    dim = len(x)
    o = np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim
    return o

def F10(x):
    dimension = len(x)
    return -20 * math.exp(-0.2 * math.sqrt(sum(xi**2 for xi in x) / dimension)) - \
            math.exp(sum(math.cos(2 * math.pi * xi) for xi in x) / dimension) + 20 + math.e

def F11(x):
    dim=len(x);
    w=[i for i in range(len(x))]
    w=[i+1 for i in w];
    o=np.sum(x**2)/4000-prod(np.cos(x/np.sqrt(w)))+1;   
    return o;

def F12(x):
    dim=len(x);
    o=(math.pi/dim)*(10*((np.sin(math.pi*(1+(x[0]+1)/4)))**2)+np.sum((((x[1:dim-1]+1)/4)**2)*(1+10*((np.sin(math.pi*(1+(x[1:dim-1]+1)/4))))**2))+((x[dim-1]+1)/4)**2)+np.sum(Ufun(x,10,100,4));   
    return o;

def F13(x): 
    dim=len(x);
    o=.1*((np.sin(3*math.pi*x[1]))**2+sum((x[0:dim-2]-1)**2*(1+(np.sin(3*math.pi*x[1:dim-1]))**2))+ 
    ((x[dim-1]-1)**2)*(1+(np.sin(2*math.pi*x[dim-1]))**2))+np.sum(Ufun(x,5,100,4));
    return o;

def F14(x):
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                    [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.zeros(25)
    v = np.matrix(x).reshape(-1, 1)  # Mengubah dimensi x menjadi (13, 1)

    for i in range(25):
        H = v - aS[:, i]
        bS[i] = np.sum((np.power(H, 6)))

    w = np.arange(1, 26)
    o = ((1.0 / 500) + np.sum(1.0 / (w + bS))) ** (-1)
    return o

def F15(L):  
    aK=[.1957,.1947,.1735,.16,.0844,.0627,.0456,.0342,.0323,.0235,.0246];
    bK=[.25,.5,1,2,4,6,8,10,12,14,16];
    aK=np.asarray(aK);
    bK=np.asarray(bK);
    bK = 1/bK;  
    fit=np.sum((aK-((L[0]*(bK**2+L[1]*bK))/(bK**2+L[2]*bK+L[3])))**2);
    return fit

def F16(L):  
    o=4*(L[0]**2)-2.1*(L[0]**4)+(L[0]**6)/3+L[0]*L[1]-4*(L[1]**2)+4*(L[1]**4);
    return o


def F17(L):
    o = (L[1] - (L[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * L[0] - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(L[0]) + 10
    return o


def F18(L):  
    o=(1+(L[0]+L[1]+1)**2*(19-14*L[0]+3*(L[0]**2)-14*L[1]+6*L[0]*L[1]+3*L[1]**2))*(30+(2*L[0]-3*L[1])**2*(18-32*L[0]+12*(L[0]**2)+48*L[1]-36*L[0]*L[1]+27*(L[1]**2)));
    return o

def F19(L):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    o = 0

    for i in range(4):
        o = o - cH[i] * np.exp(-np.sum(aH[i, :] * ((L[:, np.newaxis] - pH[i, :]) ** 2)))

    return o

def F20(L):    
    aH = [[10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]]
    aH = np.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = np.asarray(cH)
    pH = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]
    pH = np.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * np.exp(-np.sum(aH[i, :] * ((L[:6] - pH[i, :6]) ** 2)))
    return o

def F21(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    o = 0.0
    for i in range(5):
        v = np.subtract(x, aSH[i])
        try:
            inverse = 1.0 / (np.dot(v, v.T) + cSH[i])
            o = o - inverse
        except np.linalg.LinAlgError:
            continue

    return o

def F22(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    o = 0.0
    for i in range(7):
        v = np.subtract(x, aSH[i])
        try:
            inverse = 1.0 / (np.dot(v, v.T) + cSH[i])
            o = o - inverse
        except np.linalg.LinAlgError:
            continue

    return o

def F23(x):
    aSH = [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
            [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    R = 0
    for i in range(10):
        R -= ((x - aSH[i]) @ (x - aSH[i]).T + cSH[i])**(-1)
    return R

def getFunctionDetails(a):
       
    # [name, lb, ub, dim]
    param = {  0: ["F1",-100,100,30],
               1: ["F2",-10,10,30],
               2: ["F3",-100,100,30],
               3: ["F4",-100,100,30],
               4: ["F5",-30,30,30],
               5: ["F6",-100,100,30],
               6: ["F7",-1.28,1.28,30],
               7: ["F8",-500,500,30],
               8: ["F9",-5.12,5.12,30],
               9: ["F10",-32,32,30],
               10: ["F11",-600,600,30],
               11: ["F12",-50,50,30],
               12: ["F13",-50,50,30],
               13: ["F14",-65.536,65.536,2],
               14: ["F15",-5,5,4],
               15: ["F16",-5,5,2],
               16: ["F17",-5,5,2],
               17: ["F18",-2,2,2],
               18: ["F19",0,1,3],
               19: ["F20",0,1,6],
               20: ["F21",0,10,4],
               21: ["F22",0,10,4],
               22: ["F23",0,10,4],
            }
    return param.get(a, "nothing")