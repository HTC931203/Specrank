# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:00:29 2020

@author: CCRG
"""
import numpy as np
import matplotlib.pyplot as plt
#%% load data
update = np.load("dim2_3 journal con updatepara.npz")

true_w = update['update_w']
true_beta = update['update_beta']
true_beta = np.delete(true_beta, 0, axis=1)
true_x = update['update_x']
#=============================workers' capability vectors===============================
true_c = np.dot(true_beta,true_w.T)
true_c = np.argmax(true_c,axis=1)
cluster = 4
#%% Normalized each vector
true_x = true_x/np.max(np.abs(true_x))
true_beta = true_beta/np.max(np.abs(true_beta))
#===========================Slope of w(for 2 dim)================================
slope_w = true_w[:,1]/true_w[:,0]

x = np.linspace(-1,1)
#===========================Slope of u_c(for 2 dim)================================
y1 = slope_w[0]*x
y2 = slope_w[1]*x
y3 = slope_w[2]*x
y4 = slope_w[3]*x

clu = []
for c in range (cluster):
    clusterind = np.where(true_c==c)[0]
    clu.append(clusterind) 
 
C1 = true_beta[clu[0]]
C2 = true_beta[clu[1]]
C3 = true_beta[clu[2]]
C4 = true_beta[clu[3]]

plt.figure(figsize=(8,8))

arrow1 = plt.arrow(0,0,C1[0][0],C1[0][1],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='mistyrose',linestyle='dashed',label='C1 worker')
for d in range(1,C1.shape[0]):
    plt.arrow(0,0,C1[d][0],C1[d][1],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='mistyrose',linestyle='dashed')

arrow2 = plt.arrow(0,0,C2[0][0],C2[0][1],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='cyan',linestyle='dashed',label='C2 worker')  
for d in range(1,C2.shape[0]):
    plt.arrow(0,0,C2[d][0],C2[d][1],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='cyan',linestyle='dashed')
    
arrow3 = plt.arrow(0,0,C3[0][0],C3[0][1],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='lightgreen',linestyle='dashed',label='C3 worker')  
for d in range(1,C3.shape[0]):
    plt.arrow(0,0,C3[d][0],C3[d][1],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='lightgreen',linestyle='dashed')
    
arrow4 = plt.arrow(0,0,C4[0][0],C4[0][1],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='wheat',linestyle='dashed',label='C4 worker')  
for d in range(1,C3.shape[0]):
    plt.arrow(0,0,C4[d][0],C4[d][1],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='wheat',linestyle='dashed')
    
#%% true u_c vectors' distribution
arrow5 = plt.arrow(x[48],y1[48],x[1]-x[48],y1[1]-y1[48],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='lightcoral',label='C1_CoeVec')
arrow6 = plt.arrow(x[0],y2[0],x[49]-x[0],y2[49]-y2[0],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='dodgerblue',label='C2_CoeVec')
arrow7 = plt.arrow(x[2],y3[2],x[46]-x[2],y3[46]-y3[2],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='limegreen',label='C3_CoeVec')
arrow8 = plt.arrow(x[22],y4[22],x[27]-x[22],y4[27]-y4[22],width=0.001,length_includes_head=True,head_width=0.07, head_length=0.05,color='orange',label='C4_CoeVec')

plt.xlim(-1, 1)
plt.ylim(-1, 1)

my_x_ticks = np.arange(-1, 1.2, 0.2)
my_y_ticks = np.arange(-1, 1.2, 0.2)
plt.xticks(my_x_ticks,fontsize=14)
plt.yticks(my_y_ticks,fontsize=14)
plt.legend([arrow1,arrow2,arrow3,arrow4,arrow5,arrow6,arrow7,arrow8], ['Class 1 Worker Specialty Vector','Class 2 Worker Specialty Vector','Class 3 Worker Specialty Vector','Class 4 Worker Specialty Vector','Class 1 Feature Vector','Class 2 Feature Vector','Class 3 Feature Vector','Class 4 Feature Vector'],prop={'size': 10},loc='lower left')
plt.gcf().savefig("worker_project_on_journal_dataet.png",bbox_inches='tight', dpi=300)
plt.show()


