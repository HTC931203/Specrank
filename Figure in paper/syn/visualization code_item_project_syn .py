# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:58:20 2020

@author: CCRG
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
#%% load data
update = np.load("specrank_regularized reformed on missing20_cover previous miss data updatepara.npz")
true_w = np.load("true w.npy")
true_x = np.load("true x.npy")
true_c = np.load("specrank_regularized reformed on missing20_cover previous miss data update_expa.npz")['arr_0']
true_c = np.argmax(true_c,axis=1)
cluster = 2
#%% Normalized each vector
true_x = true_x/np.max(np.abs(true_x))
#===========================Slope of w(for 2 dim)================================
slope = true_w[:,1]/true_w[:,0]

x = np.linspace(-1,1)
y1 = slope[0]*x
y2 = slope[1]*x
#==============================================================================

#================================scatter=======================================
clu = []
for c in range (cluster):
    clusterind = np.where(true_c==c)[0]
    clu.append(clusterind)    

C1 = true_x[clu[0]]
C2 = true_x[clu[1]]

plt.figure(figsize=(8,8))

C1_item = plt.scatter(C1[:,0],C1[:,1],s=30,c='red',marker='o',alpha=0.3,label='C1')
C2_item = plt.scatter(C2[:,0],C2[:,1],s=50,c='blue',marker='*',alpha=0.3,label='C2') 

#==================================projection of score=================================
m1 = slope[0]
m2 = slope[1]
b1 = 0
b2 = 0
'''
If you want to plot class 1(2), choose 'arrow1(2)', and change the class in for loop condition.
'''
arrow1 = plt.arrow(x[0],y1[0],x[49]-x[0],y1[49]-y1[0],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='lightcoral',label='C1_CoeVec')
#arrow2 = plt.arrow(x[0],y2[0],x[33]-x[0],y2[33]-y2[0],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='dodgerblue',label='C2_CoeVec')

for d in range(C1.shape[0]):
    '''Class1'''    
    x0 = C1[d,0]
    y0 = C1[d,1]
    x1 = (m1*y0+x0-m1*b1)/(m1**2+1)
    y1 = (m1**2*y0+m1*x0+b1)/(m1**2+1)
    plt.scatter(x1,y1,s=30,c='red',marker='x',alpha=0.7)
    plt.plot([x0,x1],[y0,y1], 'r--')
    
#    '''Class2'''      
#    x2 = C2[d,0]
#    y2 = C2[d,1]
#    x3 = (m2*y2+x2-m2*b2)/(m2**2+1)
#    y3 = (m2**2*y2+m2*x2+b2)/(m2**2+1)
#    plt.scatter(x3,y3,s=30,c='blue',marker='x',alpha=0.7)
#    plt.plot([x2,x3],[y2,y3],'b--')    

#==============================================================================

plt.xlim(-1, 1)
plt.ylim(-1, 1)

my_x_ticks = np.arange(-1.0, 1.2, 0.2)
my_y_ticks = np.arange(-1.0, 1.2, 0.2)
plt.xticks(my_x_ticks,fontsize=14)
plt.yticks(my_y_ticks,fontsize=14)
plt.legend([C1_item,C2_item,arrow1], ['Class 1','Class 2','Class 1 Feature Vector',],prop={'size': 12},loc='upper right')
#plt.gcf().savefig("syn_true_C1_pro.png",bbox_inches='tight', dpi=300)
plt.show()     


























