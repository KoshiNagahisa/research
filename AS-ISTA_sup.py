#DARTSを実行

import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import module.makefolder as mf
import os
from matplotlib.animation import ArtistAnimation
import optuna
import main_function_torch as mft

m,n,k = 75,150,12
snr =10 
p = k/n 

sigma = np.sqrt(p/(10**(snr/10)))
lam = 10

max_itr = 100

device = torch.device('cuda') # 'cpu' or 'cuda'

# def objective(trial):

#     mbs   = trial.suggest_int('mbs',10,70) # ミニバッチサイズ
#     adam_lr1 = trial.suggest_float('adam_lr1',0.00001,0.0002) # Adamの学習率
#     adam_lr2 = trial.suggest_float('adam_lr2',0.01,1.0)
#     para_num = trial.suggest_int('para_num',500,1500)    #パラメータの更新回数
    
mbs   = 29 # ミニバッチサイズ
adam_lr1 = 0.00013680841647393453 # Adamの学習率
adam_lr2 = 0.9738576258007406
para_num = 1363  #パラメータの更新回数

file_name = os.path.splitext(os.path.basename(__file__))[0]
folder = mf.make_folder(file_name)

A = torch.normal(torch.zeros(m, n), std = 1.0).detach().numpy()
alpha_ini = 1/np.max( np.abs( np.linalg.eigvals(A.T@A) ) )*torch.ones(max_itr)

# beta1_ini =10*torch.ones(max_itr)
# beta2_ini =-10*torch.ones(max_itr)
# gamma1_ini =-10*torch.ones(max_itr)
# gamma2_ini =10*torch.ones(max_itr)

beta1_ini =torch.zeros(max_itr)
beta2_ini =torch.zeros(max_itr)
gamma1_ini =torch.zeros(max_itr)
gamma2_ini =torch.zeros(max_itr)

#学習型反復アルゴリズムのクラス定義
class ISTA(nn.Module):
    def __init__(self,max_itr):
        super(ISTA, self).__init__()
        self.alpha = nn.Parameter(alpha_ini) # 学習可能ステップサイズパラメータ
        self.beta1 = nn.Parameter(beta1_ini) #構造パラメータ
        self.beta2 = nn.Parameter(beta2_ini)
        self.gamma1 = nn.Parameter(gamma1_ini)
        self.gamma2 = nn.Parameter(gamma2_ini)
    #l1ノルムの近接写像
    def prox_L1(self, x, tau):
        return torch.sgn(x) * torch.maximum(torch.abs(x) - tau, torch.tensor((0)))

    def GD_step(self,x,tau):
        return x + tau * (y - x @ A.t()) @ A
    
    def ST_step(self, x, tau):
        return self.prox_L1(x, lam * tau)

    #アルゴリズム本体
    def forward(self,num_itr):
        s = torch.zeros(mbs, n).to(device) # 初期探索点
        for i in range(num_itr):
            r = torch.exp(self.beta1[i])/(torch.exp(self.beta1[i])+torch.exp(self.beta2[i])) *self.GD_step(s, self.alpha[i]) + torch.exp(self.beta2[i])/(torch.exp(self.beta1[i])+torch.exp(self.beta2[i])) * self.ST_step(s, self.alpha[i])
            s = torch.exp(self.gamma1[i])/(torch.exp(self.gamma1[i])+torch.exp(self.gamma2[i])) * self.GD_step(r, self.alpha[i]) + torch.exp(self.gamma2[i])/(torch.exp(self.gamma1[i])+torch.exp(self.gamma2[i]))  * self.ST_step(r, self.alpha[i])
        return s


model= ISTA(max_itr)
opt1 = optim.Adam(model.parameters(), lr=adam_lr1)
opt2 = optim.Adam(model.parameters(), lr=adam_lr2)
loss_func = nn.MSELoss()
loss_MSE = np.zeros((max_itr*2))
loss_para = np.zeros((max_itr*para_num*2))
img_list=[]
k,l = 0,0
epoch_max = 6

fig, ax = plt.subplots()
plt.yscale('log')
plt.grid(which='major',color='black',linestyle='-')
plt.grid(which='minor',color='black',linestyle='--')
for epoch in tqdm(range(epoch_max)):
    for j in range(2):
        if j == 0:
            model.alpha.requires_grad = True
            model.beta1.requires_grad = False
            model.beta2.requires_grad = False
            model.gamma1.requires_grad = False
            model.gamma2.requires_grad = False
        else:            
            model.alpha.requires_grad = False
            model.beta1.requires_grad = True
            model.beta2.requires_grad = True
            model.gamma1.requires_grad = True
            model.gamma2.requires_grad = True
        for i in range(para_num):
            x = mft.gen_minibatch(mbs,n,p).to(device) # 元信号の生成
            w = torch.normal(torch.zeros(mbs, m), sigma).to(device)
            A = torch.normal(torch.zeros(m, n), std = 1.0).to(device) # 観測行列
            y = torch.mm(x, A.t()).to(device) + w # 観測信号の生成
            opt1.zero_grad()
            opt2.zero_grad()
            x_hat = model(max_itr)
            loss = loss_func(x_hat, x)     #教師あり学習
            loss.backward()
            if j==0 : 
                opt1.step() 
                model.alpha.data.clamp_(min=0.0)
                artist = ax.plot(model.alpha.data,'blue')
                img_list.append(artist)
            else: 
                opt2.step()
                for idx in range(max_itr):
                    if model.beta1.data[idx] >= model.beta2.data[idx]:
                        model.beta1.data[idx] = 10
                        model.beta2.data[idx] = -10
                    else:
                        model.beta2.data[idx] = 10
                        model.beta1.data[idx] = -10

                    if model.gamma1.data[idx] > model.gamma2.data[idx]:
                        model.gamma1.data[idx] = 10
                        model.gamma2.data[idx] = -10
                    else:
                        model.gamma2.data[idx] = 10
                        model.gamma1.data[idx] = -10

            loss_para[k] = loss.item()
            k = k + 1
        loss_MSE[l]=loss.item()
        l=l+1
        if j == 0:  
            print("epoch",epoch+1," α learned", '{:.4e}'.format(loss.item()))
        else:  
            print("epoch",epoch+1,' β learned', '{:.4e}'.format(loss.item()))
        
#         intermediate_value = loss
#         trial.report(intermediate_value, epoch)

#         if trial.should_prune():    # trial.should_prune()はTrueならば試行が打ち切られる
#             print("epoch", epoch+1, "で打ち切り")  # 何回めのエポックで打ち切ったか見るために追加
#     return loss
# # 最適化を実行
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=5)

# # 最適解の出力
# print(f"The best value is : \n {study.best_value}")
# print(f"The best parameters are : \n {study.best_params}")

# ステップサイズをテキストファイルで記録
np.savetxt('alpha_DARTS_sup.txt',model.alpha.detach().numpy())
np.savetxt('beta1_DARTS_sup.txt',model.beta1.detach().numpy())
np.savetxt('beta2_DARTS_sup.txt',model.beta2.detach().numpy())
np.savetxt('gamma1_DARTS_sup.txt',model.gamma1.detach().numpy())
np.savetxt('gamma2_DARTS_sup.txt',model.gamma2.detach().numpy())



anim = ArtistAnimation(fig, img_list)
anim.save(folder+"/alpha_anime.gif",writer="pillow")


fig, ax = plt.subplots(constrained_layout = True)
plt.grid()
ax.set_box_aspect(1)
ax.set_xlabel('iteration',fontsize=20)
ax.tick_params(labelsize=20)
ax.plot(loss_para,color="red")
plt.yscale('log')
plt.savefig(folder+"/loss_DARTS.pdf")

fig, ax = plt.subplots(constrained_layout = True)
plt.grid()
ax.set_box_aspect(1)
ax.set_xlabel('iteration',fontsize=20)
ax.tick_params(labelsize=20)
ax.plot(loss_MSE,color="red")
plt.yscale('log')
plt.savefig(folder+"/loss_MSE_DARTS.pdf")