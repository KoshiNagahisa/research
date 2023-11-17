
#ステップサイズを入力として各層のMSEと目的関数の値のグラフを出力する

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import module.makefolder as mf
import time
import datetime as dt
import os
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 

m = 75 # 観測ベクトル次元
n = 150 # 元信号ベクトル次元
snr = 10
k = 12
p = k/n # 元信号の非ゼロ元の生起確率
sigma = np.sqrt(p/(10**(snr/10)))
mbs1   = 100# ミニバッチサイズ（xの数）
mbs2 = 100  #Aの数

lam = 10  #正則化係数

max_itr = 100 #反復回数

device = torch.device('cuda') # 'cpu' or 'cuda'

file_name = os.path.splitext(os.path.basename(__file__))[0]
folder = mf.make_folder(file_name)

#l0ノルムの近接写像
def prox_L0(x, tau):
  epsilon = 1e-10
  th = np.sqrt(2 * tau + epsilon)
  return np.where(np.abs(x) < th, 0, x)

#l(1/2)ノルムの近接写像(近似)
def prox_L12(x, tau):
    sigma = 0.1
    epsilon = 1e-10
    tau_pos = tau + epsilon
    th2 = 3 / 2 * (tau_pos ** (2/3))
    th1 = th2 * (1 - sigma)
        
    prox1 = 2 / (3 * sigma) * x + np.sign(x) * (1 - 1 / sigma) * (tau_pos ** (2/3))
    prox2 = 2 / 3 * x * (1 + np.cos(2 / 3 * np.arccos( np.clip(-3 ** (3/2) / 4 * tau_pos * ((np.abs(x)+epsilon) ** (-3/2)), a_min=-1, a_max=1) ) ))

    return np.where(np.abs(x) <= th1, 0, prox1) + np.where(np.abs(x) <= th2, 0, prox2 - prox1)

#l1ノルムの近接写像
def prox_L1(x, tau):
  return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

#L0ノルム
def l0_norm(x):
  return np.sum(np.where(np.abs(x)>0,1,0))
#L1/2ノルム
def l12_norm(x):
  return np.sum((np.abs(x)+1e-10)**(1/2))
#L1ノルム
def l1_norm(x):
  return np.sum(np.abs(x))

#ミニバッチ生成関数
def gen_minibatch():
    seq = torch.normal(torch.zeros(mbs1, n), 1.0) # ガウス乱数ベクトルの生成
    support = torch.bernoulli(p * torch.ones(mbs1, n)) # 非ゼロサポートの生成
    return seq * support # 要素ごとの積(アダマール積)になることに注意

def GD_step(x, tau):
        return x + tau * (y - x @ A.T) @ A
def S_step(x, tau):
        return prox_L1(x, lam * tau)        

# coef_GD1=np.zeros(max_itr)
# coef_GD2=np.zeros(max_itr)
# coef_ST1=np.zeros(max_itr)
# coef_ST2=np.zeros(max_itr)
#ISTAの関数(MSEを出力)
def ISTA_MSE(y,A,x,alpha,beta1,beta2,gamma1,gamma2,itr):
  s = np.zeros((mbs1, n))
  loss1 = np.zeros(itr)
  loss1[0] = (1/n)*np.sum(np.square(x-s))     
  for i in range(0,itr-1):
    r = np.exp(beta1[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * GD_step(s, alpha[i]) + np.exp(beta2[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * S_step(s, alpha[i])
    s = np.exp(gamma1[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * GD_step(r, alpha[i]) + np.exp(gamma2[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * S_step(r, alpha[i])
    loss1[i+1] = (1/n)*np.sum(np.square(x-s))     
  return loss1/mbs1

#ISTAの関数(目的関数の値を出力)
def ISTA_Fy(y,A,x,alpha,beta1,beta2,gamma1,gamma2,itr):
  s = np.zeros((mbs1, n))
  loss2 = np.zeros(itr)
  loss2[0] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)
  for i in range(itr):
    r = np.exp(beta1[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * GD_step(s, alpha[i]) + np.exp(beta2[i])/(np.exp(beta1[i])+np.exp(beta2[i])) * S_step(s, alpha[i])
    s = np.exp(gamma1[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * GD_step(r, alpha[i]) + np.exp(gamma2[i])/(np.exp(gamma1[i])+np.exp(gamma2[i])) * S_step(r, alpha[i])
    loss2[i] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)
  return loss2/mbs1


def vanilla_ISTA_MSE(y,A,x,alpha,itr):
  s = np.zeros((mbs1, n))
  loss1 = np.zeros(itr)
  loss1[0] = (1/n)*np.sum(np.square(x-s))     
  if np.size(alpha) == 1:
    alpha = np.array(alpha* np.ones(itr))
  for i in range(0,itr-1):
    r =  GD_step(s, alpha[i])
    s =  S_step(r, alpha[i])
    loss1[i+1] = (1/n)*np.sum(np.square(x-s))     
  return loss1/mbs1

def vanilla_ISTA_Fy(y,A,x,alpha,itr):
  s = np.zeros((mbs1, n))
  loss2 = np.zeros(itr)
  loss2[0] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)    
  if np.size(alpha) == 1:
    alpha = np.array(alpha* np.ones(itr))     
  for i in range(itr):
    r =  GD_step(s, alpha[i])
    s =  S_step(r, alpha[i])
    loss2[i] = (1/2) * np.sum(np.square((y - s@A.T))) + lam * l1_norm(s)    
  return loss2/mbs1



loss1_test = np.zeros(max_itr)
loss2_test = np.zeros(max_itr)
loss1_vanilla = np.zeros(max_itr)
loss2_vanilla = np.zeros(max_itr)
loss1_L_s = np.zeros(max_itr)
loss2_L_s = np.zeros(max_itr)

for i in tqdm(range(mbs2)):

  x = gen_minibatch().detach().numpy()
  w = torch.normal(torch.zeros(mbs1, m), sigma).detach().numpy()
  A = torch.normal(torch.zeros(m, n), std = 1.0).detach().numpy() # 観測行列
  y = x@A.T + w
 
  alpha_ini = 1/np.max( np.abs( np.linalg.eigvals(A.T@A) ) )


  #通常のISTA# 
  loss1_vanilla += vanilla_ISTA_MSE(y,A,x,alpha_ini, max_itr)
  loss2_vanilla += vanilla_ISTA_Fy(y,A,x,alpha_ini, max_itr)

  #DARTS-ISTA
  alpha_test = np.loadtxt('alpha_DARTS_exp1.txt')
  beta1_test = np.loadtxt('beta1_DARTS_exp1.txt')
  beta2_test = np.loadtxt('beta2_DARTS_exp1.txt')
  gamma1_test = np.loadtxt('gamma1_DARTS_exp1.txt')
  gamma2_test = np.loadtxt('gamma2_DARTS_exp1.txt')
  # lam_test = np.loadtxt('lam_DARTS_exp1.txt')

  loss1_test += ISTA_MSE(y,A,x,alpha_test, beta1_test,beta2_test,gamma1_test,gamma2_test,max_itr)
  loss2_test += ISTA_Fy(y,A,x,alpha_test, beta1_test,beta2_test,gamma1_test,gamma2_test,max_itr)
 
  #ステップサイズのみ学習したLISTA
  beta_L_s = np.loadtxt('step_size.txt')
  loss1_L_s += vanilla_ISTA_MSE(y,A,x,beta_L_s,max_itr)
  loss2_L_s += vanilla_ISTA_Fy(y,A,x,beta_L_s,max_itr)

loss1_vanilla /= mbs2
loss1_test/= mbs2
loss1_L_s /= mbs2

loss2_vanilla /= mbs2
loss2_test /= mbs2
loss2_L_s /= mbs2

#MSEのグラフ出力
fig, ax = plt.subplots(constrained_layout = True)
x_max = max_itr
plt.grid(which='major',color='black',linestyle='-')
plt.grid(which='minor',color='gray',linestyle='--')
plt.xlim(0,x_max)
plt.xticks([x_max*(0/5),x_max*(1/5),x_max*(2/5),x_max*(3/5),x_max*(4/5),x_max*(5/5)])
#ax.set_box_aspect(1)

ax.set_xlabel('iteration t',fontsize=20)
ax.set_ylabel('MSE',fontsize=20)
ax.tick_params(labelsize=20)

ax.plot(loss1_vanilla,color = '#000000',linestyle = ":",label = 'ISTA',marker='s',markevery = 15)
ax.plot(loss1_L_s,color = 'orange',linestyle = "--",label = 'ISTA(Learned γ)',marker='o',markevery = 15)
ax.plot(loss1_test,color = 'red',label = 'AS-ISTA',marker='^',markevery = 15)


plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.tight_layout()
plt.yscale('log')
plt.savefig(folder+"/DARTS_MSE_L1_10.pdf")

#目的関数の値のグラフ出力
fig, ax = plt.subplots(constrained_layout = True)

plt.grid(which='major',color='black',linestyle='-')
plt.grid(which='minor',color='gray',linestyle='--')
plt.xlim(0,max_itr)
plt.xticks([x_max*(0/5),x_max*(1/5),x_max*(2/5),x_max*(3/5),x_max*(4/5),x_max*(5/5)])
#ax.set_box_aspect(1)

ax.set_xlabel('iteration t',fontsize=20)
ax.set_ylabel('objective function',fontsize=20)
ax.tick_params(labelsize=20)

ax.plot(loss2_vanilla,color = '#000000',linestyle = ":",label = 'ISTA',marker='s',markevery = 15)
ax.plot(loss2_L_s,color = 'orange',linestyle = "--",label = 'ISTA(Learned γ)',marker='o',markevery = 15)
ax.plot(loss2_test,color = 'red',label = 'AS-ISTA',marker='o',markevery = 15)


plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.tight_layout()
plt.yscale('log')
plt.savefig(folder+"/DARTS_Fy_L1_10.pdf")

#各反復でのステップサイズ
lip = 0
for gen in tqdm(range(max_itr)):
    A = torch.normal(torch.zeros(m, n), std = 1.0).to(device) # 観測行列
    D = A.detach().cpu().numpy()
    lip += 1/np.max( np.abs( np.linalg.eigvals(D.T@D) ) )
lip /= max_itr
lip = lip*np.ones(max_itr)

fig, ax = plt.subplots(constrained_layout = True)
plt.grid()
plt.xlim(0,max_itr)
plt.xticks([0,20,40,60,80,100])
#ax.set_box_aspect(1)
ax.set_xlabel('iteration t',fontsize=20)
ax.set_ylabel('step size',fontsize=20)
ax.tick_params(labelsize=20)
plt.yscale('log')

ax.plot(range(max_itr),lip,color="k",linestyle = ":",label = 'ISTA')
ax.plot(beta_L_s,color="orange",linestyle = "--",label = 'ISTA(Learned γ)')
ax.plot(alpha_test,color="red",label = 'AS-ISTA')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)



plt.savefig(folder+"/alpha.pdf")


# fig, ax = plt.subplots(constrained_layout = True)
# plt.grid()
# plt.xlim(0,max_itr)
# plt.xticks([0,20,40,60,80,100])

# ax.set_xlabel('iteration t',fontsize=20)
# ax.set_ylabel('step size',fontsize=20)
# ax.tick_params(labelsize=20)
# plt.yscale('log')

# ax.plot(lam_test,color="red",label = 'AS-ISTA')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)

# plt.savefig(folder+"/lam.pdf")

GD_coef1 = np.exp(beta1_test)/(np.exp(beta1_test)+np.exp(beta2_test))
ST_coef1 = np.exp(beta2_test)/(np.exp(beta1_test)+np.exp(beta2_test))

fig, ax = plt.subplots(constrained_layout = True)
plt.grid()
plt.xlim(0,max_itr)
plt.xticks([0,20,40,60,80,100])
ax.set_box_aspect(1)
ax.set_xlabel('iteration t',fontsize=20)
ax.set_ylabel('coefficient of operation',fontsize=20)
ax.tick_params(labelsize=20)
ax.plot(GD_coef1,color="green",label='wr1')
ax.plot(ST_coef1,color="orange",label='wr2')
ax.legend(prop={'size':12,})
plt.savefig(folder+"/coefficient1.png")



GD_coef2 = np.exp(gamma1_test)/(np.exp(gamma1_test)+np.exp(gamma2_test))
ST_coef2 = np.exp(gamma2_test)/(np.exp(gamma1_test)+np.exp(gamma2_test))

fig, ax = plt.subplots(constrained_layout = True)
plt.grid()
plt.xlim(0,max_itr)

plt.xticks([0,20,40,60,80,100])
ax.set_box_aspect(1)
ax.set_xlabel('iteration t',fontsize=20)
ax.set_ylabel('coefficient of operation',fontsize=20)
ax.tick_params(labelsize=20)
ax.plot(GD_coef2,color="green",label='wx1')
ax.plot(ST_coef2,color="orange",label='wx2')
ax.legend(prop={'size':12,})
plt.savefig(folder+"/coefficient2.png")



Architecture = np.zeros(2*max_itr)

for i in range(2*max_itr):
  if i % 2== 0: Architecture[i]=GD_coef1[int(i/2)]
  else: Architecture[i]=GD_coef2[int(i/2)]

np.savetxt('Architecture.txt',Architecture)
np.savetxt('MSE1.txt',loss1_test)
np.savetxt('MSE2.txt',loss1_L_s)


Architecture = np.loadtxt('Architecture.txt')
arr_2d = Architecture.reshape((10,20))
df = pd.DataFrame(data=arr_2d, index=[i for i in range(1,11)], columns=[j for j in range(1, 21)])
fig, ax = plt.subplots(constrained_layout = True)
ax.set_box_aspect(1)
sns.heatmap(df, linewidths=1 ,cmap='Greens',vmin=-0.19,vmax=1.19)
plt.savefig(folder+"/Architecture.pdf")

Architecture = np.loadtxt('MSE1.txt')
arr_2d = Architecture.reshape((10,10))
df = pd.DataFrame(data=arr_2d, index=[i for i in range(1,11)], columns=[j for j in range(1, 11)])
fig, ax = plt.subplots(constrained_layout = True)
ax.set_box_aspect(1)
sns.heatmap(df, linewidths=1 ,cmap='rainbow',norm=LogNorm(vmin=df.min().min(), vmax=df.max().max()))
plt.savefig(folder+"/MSE1.pdf")

array = np.zeros(2*max_itr)
for i in range(2*max_itr):
  if i%2 == 0:
    array[i] = 1

arr_2d = array.reshape((10,20))
df = pd.DataFrame(data=arr_2d, index=[i for i in range(1,11)], columns=[j for j in range(1, 21)])
fig, ax = plt.subplots(constrained_layout = True)
ax.set_box_aspect(1)
sns.heatmap(df, linewidths=1 ,cmap='Greens',vmin=-0.19,vmax=1.19)
plt.savefig(folder+"/Architecture_ISTA.pdf")
