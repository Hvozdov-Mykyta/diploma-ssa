
import os
import numpy as np
import array as arr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# оголошення за задання необхідних змінних
hiddenSize =28 # кількість нейронів на прихованому шарі
num =7 # кількість шарів
alpha=0.0001
eps=0.00001
arr_eps = [] # масив для значень похибки навчання
arr_age = [] # масив для значень кількості епох навчання

def sigmoid(x): # функція активації
    c=1
    return 1 / (1 + np.exp(-x*c))
    #return 1+0.5*np.arctan(x)
    
    #return c*x
    #return (np.exp(c*x)-np.exp(-c*x)) / (np.exp(c*x) + np.exp(-c*x))
def sigmoid_output_to_derivative(output): # метод обчислення похідної від функції активації
    c=1
    #return c
    return output*c*(1 - output)
    #return 1 - output**2
def gen_synapse(x, y, hiddenSize, num): # генерація початкових ваг
    synapse = []
    np.random.seed(1)

    for i in range(num):
        if i == 0:
            synapse.append(2 * np.random.random((len(x[0]),hiddenSize)) - 1)
        elif i == num - 1:
            synapse.append(2 * np.random.random((hiddenSize,len(y[0]))) - 1)
        else:
            synapse.append(2 * np.random.random((hiddenSize,hiddenSize)) - 1)
    return synapse
    


Numt1=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1]
Numt2=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
Numt3=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

Num81=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
Num82=[1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
Num83=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,0]
Num84=[1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
Num85=[1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1]

Num91=[1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]
Num92=[1,0,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]
Num93=[1,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]
Num94=[1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0]
Num95=[1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0]

Num01=[1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num02=[1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num03=[1,1,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num04=[1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,1,1,1]
Num05=[1,1,1,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1]

Num21=[1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
Num22=[1,1,1,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
Num23=[1,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
Num24=[1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0]
Num25=[1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1]

Num31=[1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num32=[1,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num33=[1,1,1,1,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num34=[1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1]
Num35=[1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0]

Num41=[1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num42=[0,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num43=[1,0,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num44=[1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0]
Num45=[1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1]

Num61=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num62=[0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1]
Num63=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,0]
Num64=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,1]
Num65=[0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1]

Num71=[1,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0]
Num72=[0,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0]
Num73=[1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0]
Num74=[1,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0]
Num75=[1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0]


Num11=[0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num12=[0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num13=[0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]
Num14=[0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0]
Num15=[0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1]

Num51=[1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num52=[1,1,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num53=[0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num54=[1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
Num55=[1,1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,1,1,1,1]
#Num56=[1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1]
#Num57=[1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0]
             
NumEpt= [ Numt1, Numt2, Numt3]
#NumEpt= [ Numt1]
Num0X=[Num01,Num02, Num03,Num04,Num05]
Num2X=[Num21,Num22, Num23,Num24,Num25]
Num3X=[Num31,Num32, Num33,Num34,Num35]
Num4X=[Num41,Num42, Num43,Num44,Num45]
Num6X=[Num61,Num62, Num63,Num64,Num65]
Num7X=[Num71,Num72, Num73,Num74,Num75]
Num8X=[Num81,Num82, Num83,Num84,Num85]
Num9X=[Num91,Num92, Num93,Num94,Num95]          
Num1X=[Num11,Num12, Num13,Num14,Num15]
Num5X=[Num51,Num52,Num53, Num54, Num55]
XX=np.array([Num0X,Num1X,Num2X,Num3X,Num4X,Num5X,Num6X,Num7X,Num8X,Num9X])

NumEpY=[[1],[1],[1]]
#NumEpY=[[1]]
Num0Y=[[1],[1],[1],[1],[1]]
Num1Y=[[1],[1],[1],[1],[1]]
Num2Y=[[1],[1],[1],[1],[1]]
Num3Y=[[1],[1],[1],[1],[1]]
Num4Y=[[1],[1],[1],[1],[1]]
Num5Y=[[1],[1],[1],[1],[1]]
Num6Y=[[1],[1],[1],[1],[1]]
Num7Y=[[1],[1],[1],[1],[1]]
Num8Y=[[1],[1],[1],[1],[1]]
Num9Y=[[1],[1],[1],[1],[1]]
YY=np.array([Num0Y,Num1Y,Num2Y,Num3Y,Num4Y,Num5Y,Num6Y,Num7Y,Num8Y,Num9Y])
X00=[];X0=X00+NumEpt
for znaX0 in XX[0]: X0.append(znaX0)

X11=[];X1=X11+NumEpt
for znaX1 in XX[1]: X1.append(znaX1)

X22=[];X2=X22+NumEpt
for znaX2 in XX[2]: X2.append(znaX2)

X33=[];X3=X33+NumEpt
for znaX3 in XX[3]: X3.append(znaX3)

X44=[];X4=X44+NumEpt
for znaX4 in XX[4]: X4.append(znaX4)

X55=[];X5=X55+NumEpt
for znaX5 in XX[5]: X5.append(znaX5)

X66=[];X6=X66+NumEpt
for znaX6 in XX[6]: X6.append(znaX6)

X77=[];X7=X77+NumEpt
for znaX7 in XX[7]: X7.append(znaX7)

X88=[];X8=X88+NumEpt
for znaX8 in XX[8]: X8.append(znaX8)

X99=[];X9=X99+NumEpt
for znaX9 in XX[9]: X9.append(znaX9)

#xx=np.array([X0,X1,X2,X3,X4,X5,X6,X7,X8,X9])
xx=np.array([Num0X,Num1X,Num2X,Num3X,Num4X,Num5X,Num6X,Num7X,Num8X,Num9X])
#print('testov_masuv_X=',xx, len(xx[0]),len(xx),type(xx))

Y00=[];Y0=Y00+NumEpY
for znaY0 in YY[0]: Y0.append(znaY0)

Y11=[];Y1=Y11+NumEpY
for znaY1 in YY[1]: Y1.append(znaY1)

Y22=[];Y2=Y22+NumEpY
for znaY2 in YY[2]: Y2.append(znaY2)

Y33=[];Y3=Y33+NumEpY
for znaY3 in YY[3]: Y3.append(znaY3)

Y44=[];Y4=Y44+NumEpY
for znaY4 in YY[4]: Y4.append(znaY4)

#print(Y1)
Y55=[];Y5=Y55+NumEpY
for znaY5 in YY[5]: Y5.append(znaY5)

Y66=[];Y6=Y66+NumEpY
for znaY6 in YY[6]: Y6.append(znaY6)

Y77=[];Y7=Y77+NumEpY
for znaY7 in YY[7]: Y7.append(znaY7)

Y88=[];Y8=Y88+NumEpY
for znaY8 in YY[8]: Y8.append(znaY8)

Y99=[];Y9=Y99+NumEpY
for znaY9 in YY[9]: Y9.append(znaY9)
#print(Y5)
#yy=np.array([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9])
yy=np.array([Num0Y,Num1Y,Num2Y,Num3Y,Num4Y,Num5Y,Num6Y,Num7Y,Num8Y,Num9Y])
#print('testov_masuv_Y=', yy,type(yy))
nn=len(X0)
mm=len(xx)
krok=0.001

iteracz=100
alpha_max=1.104
alpha_min=0.001
vubirka=5
k=2.0
print('цифр=',mm, 'krok=',krok ,'iteracz=',iteracz)
kilkist=arr.array('d',[])
chastota=arr.array('d',[])   
delta_n=arr.array('d',[])              
delta=arr.array('d',[])
deltas=arr.array('d',[])
XXX=arr.array('d',[])
xxx=arr.array('d',[])
arr_age=arr.array('d',[])
arr_eps=arr.array('d',[])
alpha_x=arr.array('d',[])
epsilon=arr.array('d',[])
kilkist_alpha=arr.array('d',[])
layer_errorss=arr.array('d',[])
iteracz_i=arr.array('d',[])
iteracz_ierr=arr.array('d',[])
beta_i=arr.array('d',[])
chvud=arr.array('d',[])
for j in range(len(xx)):
    x=np.array(xx[j])
    y=np.array(yy[j])
    print('x=',x,'y=',y,len(y), len(x), type(y),'цифра=',j)
    for alpha in np.arange(alpha_min,alpha_max,krok):
        def training(x, y, alpha, eps,layer_errorss, hiddenSize, synapse,vubirka, num, arr_age, arr_eps, alpha_x, chastota, xx, delta_n, xxx, epsilon,iteracz_i, kilkist,kilkist_alpha): # метод навчання нейронної мережі
        
        
            delta=[]
            #print("Навчання нейронної мережі")
            age = 1
            while True:
                age += 1
                layers = []
                for i in range (num + 1):
                    if i == 0:
                        layers.append(x)
                    else:
                        layers.append(sigmoid(np.dot(layers[i - 1],synapse[i - 1])))

                layer_errors = []
                layer_deltas = []
                beta=[]
                v=[]
                vhat=[]
                
                layer_errors.append(layers[num] - y)
                #print('layer_errors',layer_errors,layer_errors[0][0],len(layer_errors), 'age',age,'alpha',alpha)
                e = np.mean((layer_errors[0][1]))
                layer_errorss.append(layer_errors[0][1])
                iteracz_ierr.append(age)
                
                #layer_errorss=np.append(layer_errorss,[layer_errors])
                #print('layer_errorss',layer_errorss,len(layer_errorss), 'age',age,'alpha',alpha)
                if (age % 1) == 0:
                    arr_age.append(age)
                    arr_eps.append(e)
                    alpha_x.append(alpha)
                    #print("Похибка на " + str(age) + " ітерації: " + str(e))'''
                    #iteracz_i.append(age)
                if(age >iteracz):
                    break
                '''
                if (e < eps):
                    #print("Точність " + str(round(e, 4)) + " досягнута за " + str(age) + " епох(и)")
                    break'''
               
                layer_deltas.append(layer_errors[0] * sigmoid_output_to_derivative(layers[num]))
                beta=age/(age+3)
                
                v.append((beta**k)*((1.0 - alpha*beta)*0- alpha*layer_deltas[0]))
                vhat=np.array(v)
                #print(beta,v, layer_deltas[0], vhat*(alpha*-(alpha**2)*beta))
                d=alpha*(1-alpha*beta)*vhat-(alpha**2)*layer_deltas[0]
                #print('d=',d)
                d.shape=(vubirka,1)
                #print(d)
                d=[d]* int(hiddenSize)
                
                #print('d0',len(d),d)
                beta_i.append(beta)
                for i in range (num - 1):
                    layer_errors.append(layer_deltas[i].dot(synapse[num - 1 - i].T))
                    layer_deltas.append(layer_errors[i + 1] * sigmoid_output_to_derivative(layers[num - 1 - i]))
                    layer_deltass=layer_errors[i + 1] * sigmoid_output_to_derivative(layers[num - 1 - i])
                    
                    beta=age/(age+3)                   
                    vv = (beta**k)*((1.0 - alpha*beta)*v[i-1]- alpha*layer_deltass)
                    v.append(vv)
                    dd= alpha*(1-alpha*beta)*v[i-1]-(alpha**2)*layer_deltass
                    
                    d.append(dd)                                    
                    #print('chvud',chvud)
                                        
                    #print(len(layer_deltas),len(layers[num - 1 - i]))
                for i in range (num):
                    synapse[num - 1 - i] += layers[num - 1 - i].T.dot(d[i])
                    #print('np.mean(synapse[num - 1 - i])'+str(synapse[num - 1 - i])+'='+str(np.mean(synapse[num - 1 ])),'j=',j,'i=',i)
                    
                    #delta.append(np.mean(synapse[num - 1])) #-попередній для побудови діаграми розгалудження
                    #delta.append(np.array(layer_errorss, dtype=float))
                    deltas.append(synapse[num - 1 - i][0][0])
                    #print('delta',delta)
                    #print(np.mean(synapse[num - 1 - i]))
                    iteracz_i.append(age)
                    delta.append(e)
                    chvud.append(v[0][0])
                #print('layer_errorss',layer_errorss,len(layer_errorss),'age',age,'alpha',alpha)
                #chastota.append(alpha-delta[age-1]-delta[age-1]*delta[age-1])
                #XX.append(delta[age])
                #beta_i.append(beta)
            kilkist_alpha.append(alpha)
            #kilkist_chastota.append(np.log(len(xx)/alpha))
            #print('min(delta),max(delta)=', min(delta),max(delta))
            data=(min(delta)+max(delta))/2.0
            for delta0 in delta:
                delta0=data
                delta0=alpha-delta0-delta0*delta0    
                delta_n.append(delta0)
                #print(len(delta_n))
                xxx.append(alpha)
                epsilon.append(e)
                #iteracz_i.append(age)
                #print('data0',data0,alpha)
                data=delta0
            kilkist.append((len(delta_n)))
            
            #print(kilkist)
        #print(layers,'j=',j,'i=',i)
        synapse = gen_synapse(x, y, hiddenSize, num)        
        training(x, y, alpha, eps,layer_errorss, hiddenSize, synapse,vubirka, num, arr_age, arr_eps, alpha_x, chastota, XXX, delta_n, xxx, epsilon,iteracz_i, kilkist,kilkist_alpha)
delta_n=np.array(delta_n)
#print('len(delta_n)',len(delta_n))
chvud=np.array(chvud)
delta=np.array(delta)
deltas=np.array(deltas)
kilkist=np.array(kilkist)
arr_age=np.array(arr_age)
arr_eps=np.array(arr_eps)
chastota=np.array(chastota)
alpha_x=np.array(alpha_x)
XXX=np.array(xx)
beta_i=np.array(beta_i)
layer_errorss=np.array(layer_errorss)
#print('layer_errorss',layer_errorss,len(layer_errorss))
xxx=np.array(xxx)
epsilon=np.array(epsilon)
iteracz_i=np.array(iteracz_i)
iteracz_ierr=np.array(iteracz_ierr)
#x=np.arange(0.01,1.01,0.001)
kilkist_alpha=np.array(kilkist_alpha)
#print('epsilon',epsilon,'len(epsilon)',len(epsilon))
#Визначення спектру частот похибки
#та визначення оптимальної швидкості навчання
nm=int(((len(arr_age)-len(kilkist_alpha))*num)/mm)
print('len(deltas)', len(deltas),'len(xxx)',len(xxx),'len(arr_age)',len(arr_age),'len(kilkist_alpha)',len(kilkist_alpha),nm, 'len(XXX)',len(XXX),'len(chastota)',len(chastota))
delta_n.shape=mm,nm
alphas=[]
for alpha in np.arange(alpha_min,alpha_max,krok):
    alphas.append(alpha)
#np.save('delta_n.npy', delta_n)
print('mm',mm,'nm',nm,len(iteracz_i),len(epsilon),len(alphas))
xxx.shape=mm,nm
epsilon.shape=mm,nm
iteracz_i.shape=mm,nm
#layer_errorss.shape=mm,nm
layer_errorss_i=layer_errorss.reshape(mm,len(alphas),iteracz)
iteracz_ierri=iteracz_ierr.reshape(mm,len(alphas),iteracz)
#chvud_i=chvud.reshape(mm,len(alphas),iteracz)
beta_ii=beta_i.reshape(mm,len(alphas),iteracz-1)
#print('layer_errorss_i',layer_errorss_i,len(layer_errorss_i))
iteracz_ii=iteracz_i.reshape(mm,len(alphas),iteracz-1,num)
chvud_i=chvud.reshape(mm,len(alphas),iteracz-1,num)
deltass=deltas.reshape(mm,len(alphas),iteracz-1,num)
epsilon_i=epsilon.reshape(mm,len(alphas),iteracz-1,num)
delta_nn=delta_n.reshape(mm,len(alphas),iteracz-1,num)
xxx_i=xxx.reshape(mm,len(alphas),iteracz-1,num)
print('mm',mm,'nm',nm,epsilon.shape,iteracz_i.shape)
iteracz_i_navchan=[]
epsilon_navchan=[]
#print(iteracz_i)
alpha_optumalne_nav=[]
for ii in range(mm):
    
    data =delta_n[ii]
    #data =epsilon_i[ii][355,:,num-1]
    ps =np.abs(np.fft.fft(data)) 
    time_step =1.0
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs) 
    garm=max(ps[int(nm/8):int(7*nm/8)])
    #print(ps[int(nm/8):int(7*nm/8)])
    mnn=list(ps).index(garm)
    #print ('max_ps',garm, 'mnn=',mnn, delta_n[ii][mnn])
    alpha_optumalne_nav=[]
    if delta_n[ii][mnn]>0.01:
        alpha_optumalne=xxx[ii][mnn]
        epsilon_min=abs(epsilon[ii][mnn])
        print( 'цифра=',ii,';', 'alpha оптимальне=',round(alpha_optumalne,9),';', 'Похибка навчання=',round(epsilon_min, 9))
        alpha_optumalne_nav.append(alpha_optumalne)
        
        A=('цифра=',ii, 'alpha оптимальне=',round(alpha_optumalne,9), 'Похибка навчання=',round(epsilon_min, 9))
        np.save('alpha_optum_poxub.npy',A)
        print(np.load('alpha_optum_poxub.npy'))
        
    else: 
        print('відсутність процесу навчання')
    
    #mmm=np.where(xxx[ii][mm] == alpha_optumalne_nav[len(alpha_optumalne_nav)-1])    
    mmm=400
    plt.plot(freqs[idx], ps[idx])
    plt.title("Фур'є спектр")
    plt.xlabel(u"ω")
    plt.ylabel(u"S(ω)")
    fig = plt.gcf()
    plt.savefig("2025"+'Цифра='+str(ii) +'Вибір_4х7_SSA_5'+'num='+str(num)+ 'Neirn='+str(hiddenSize)+'Фур_є спектр'+'alpha='+str(alpha)+'krok='+str(krok)+'iter='+str(iteracz)+'iter_.png' ,dpi=300)
    plt.clf()
    #plt.show()
    
    #print(len(xxx),len(delta_n),len(kilkist), len(x))
    #plt.scatter(xxx[ii], delta_n[ii], s=1.0, alpha=0.9)
    plt.scatter(xxx_i[ii][:,:,num-1], delta_nn[ii][:,:,num-1], s=1.0, alpha=0.9)
    plt.title("Діаграма розгалуження")
    plt.xlabel(u"alpha")
    plt.ylabel(u"x")
    fig = plt.gcf()
    plt.savefig("2025"+'Цифра='+str(ii) +'Вибір_4х7_SSA_5'+'num='+str(num)+ 'Neirn='+str(hiddenSize)+'Діаграма розгалуження'+'alpha='+str(alpha)+'krok='+str(krok)+'iter='+str(iteracz)+'iter_.png' ,dpi=300)
    plt.clf()
    #plt.show()
    
    
    #plt.scatter(iteracz_ii[ii][50,:,num-1], delta_nn[ii][50,:,num-1], s=1.0, alpha=0.9)
    #plt.scatter(iteracz_ii[ii][2,:,num-1], epsilon_i[ii][2,:,num-2], s=1.0, alpha=0.9)
    plt.scatter(iteracz_ierri[ii][mmm,:], np.abs(layer_errorss_i[ii][mmm,:]), s=17.0, alpha=0.9)
    #plt.scatter(iteracz_i[ii], epsilon[ii], s=1.0, alpha=0.9)
    plt.title("Похибка навчання")
    plt.xlabel(u"Кількість ітерацій")
    plt.ylabel(u"Похибка")
    fig = plt.gcf()
    plt.savefig("2025"+'Цифра='+str(ii) +'Вибір_4х7_SSA_5'+'num='+str(num)+ 'Neirn='+str(hiddenSize)+'Похибка навчання від Кількість ітерацій'+'alpha='+str(alpha)+'krok='+str(krok)+'iter='+str(iteracz)+'iter_.png' ,dpi=300)
    plt.clf()    
    #plt.show()
    
    plt.scatter(beta_ii[ii][mmm,:], np.abs(layer_errorss_i[ii][mmm,0:iteracz-1]), s=17.0, alpha=0.9)
    plt.title("Похибка навчання")
    plt.xlabel(u"Величина інерційного параметра")
    plt.ylabel(u"Похибка")
    fig = plt.gcf()
    plt.savefig("2025"+'Цифра='+str(ii) +'Вибір_4х7_SSA_5'+'num='+str(num)+ 'Neirn='+str(hiddenSize)+'Похибка навчання від Величина інерційного параметра'+'alpha='+str(alpha)+'krok='+str(krok)+'iter='+str(iteracz)+'iter_.png' ,dpi=300)
    plt.clf()
    #plt.show()
    
    
    plt.scatter(chvud_i[ii][mmm,:,num-1], np.abs(layer_errorss_i[ii][mmm,0:iteracz-1]), s=17.0, alpha=0.9)
    plt.title("Похибка навчання від швидкості навчання")
    plt.xlabel(u"Величина швидкості навчання ")
    plt.ylabel(u"Похибка")
    fig = plt.gcf()
    plt.savefig("2025"+'Цифра='+str(ii) +'Вибір_4х7_SSA_5'+'num='+str(num)+ 'Neirn='+str(hiddenSize)+'Похибка навчання від швидкості навчання'+'alpha='+str(alpha)+'krok='+str(krok)+'iter='+str(iteracz)+'iter_.png' ,dpi=300)
    plt.clf()
    #plt.show()
    