# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:58:15 2020

@author: Howard
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:58:20 2020

@author: CCRG
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
#%% load data
update = np.load("dim2_3 journal con updatepara.npz")
true_x = update['update_x']
true_w = update['update_w']
#true_w = np.load("true w.npy")
#true_x = np.load("true x.npy")
true_c = np.load("dim2_3 journal con update_expa.npy")
true_c = np.argmax(true_c,axis=1)
cluster = 4

#%% Normalized each vector
true_x = true_x/np.max(np.abs(true_x))
#===========================Slope of w(for 2 dim)================================
slope = true_w[:,1]/true_w[:,0]

x = np.linspace(-1,1)
y1 = slope[0]*x
y2 = slope[1]*x
y3 = slope[2]*x
y4 = slope[3]*x
#==============================================================================

#================================scatter=======================================
clu = []
for c in range (cluster):
    clusterind = np.where(true_c==c)[0]
    clu.append(clusterind)    

C1 = true_x[clu[0]]
C2 = true_x[clu[1]]
C3 = true_x[clu[2]]
C4 = true_x[clu[3]]

plt.figure(figsize=(8,8))

C1_item = plt.scatter(C1[:,0],C1[:,1],s=30,c='red',marker='o',alpha=0.3,label='C1')
C2_item = plt.scatter(C2[:,0],C2[:,1],s=50,c='blue',marker='*',alpha=0.3,label='C2') 
C3_item = plt.scatter(C3[:,0],C3[:,1],s=60,c='green',marker='2',alpha=0.3,label='C3')
C4_item = plt.scatter(C4[:,0],C4[:,1],s=50,c='darkorange',marker='+',alpha=0.3,label='C4') 



#==================================projection of score=================================
m1 = slope[0]
m2 = slope[1]
m3 = slope[2]
m4 = slope[3]
b1 = 0
b2 = 0
b3 = 0
b4 = 0
'''
If you want to plot class 1(2,3,4), choose 'arrow1(2,3,4)', and change the class in for loop condition.
'''
arrow1 = plt.arrow(x[49],y1[49],x[1]-x[49],y1[1]-y1[49],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='lightcoral',label='C1_CoeVec')
#arrow2 = plt.arrow(x[0],y2[0],x[49]-x[0],y2[49]-y2[0],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='dodgerblue',label='C2_CoeVec')
#arrow3 = plt.arrow(x[0],y3[0],x[47]-x[0],y3[47]-y3[0],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='limegreen',label='C3_CoeVec')
#arrow4 = plt.arrow(x[22],y4[22],x[27]-x[22],y4[27]-y4[22],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='orange',label='C3_CoeVec')

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

#    '''Class3'''
#    x4 = C3[d,0]
#    y4 = C3[d,1]
#    x5 = (m3*y4+x4-m3*b3)/(m3**2+1)
#    y5 = (m3**2*y4+m3*x4+b3)/(m3**2+1)
#    plt.scatter(x3,y3,s=30,c='blue',marker='x',alpha=0.7)    
#    plt.plot([x4,x5],[y4,y5], 'g--')
   
#   ''' Class4'''
#    x6 = C4[d,0]
#    y6 = C4[d,1]
#    x7 = (m4*y6+x6-m4*b4)/(m4**2+1)
#    y7 = (m4**2*y6+m4*x6+b4)/(m4**2+1)    
#    plt.scatter(x7,y7,s=30,c='orange',marker='x',alpha=0.7)    
#    plt.plot([x6,x7],[y6,y7],color='darkorange',linestyle = 'dashed')   

#==============================================================================

plt.xlim(-1, 1)
plt.ylim(-1, 1)

my_x_ticks = np.arange(-1.0, 1.2, 0.2)
my_y_ticks = np.arange(-1.0, 1.2, 0.2)
plt.xticks(my_x_ticks,fontsize=14)
plt.yticks(my_y_ticks,fontsize=14)
plt.legend([C1_item,C2_item,C3_item,C4_item,arrow1], ['Class 1','Class 2','Class 3','Class 4','Class 1 Feature Vector',],prop={'size': 12},loc='lower left')
#plt.gcf().savefig("jou_es_C1_pro.png",bbox_inches='tight', dpi=300)
plt.show()     

#==============================================================================


























