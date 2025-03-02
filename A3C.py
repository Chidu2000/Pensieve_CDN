import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Network import (ActorNetwork,CriticNetwork,CDN_Select_NN)

PATH='./results/'

RAND_RANGE=1000

class A3C(object):
    def __init__(self,is_central,model_type,s_dim,action_dim,actor_lr=1e-4,critic_lr=1e-3):
        self.s_dim=s_dim
        self.a_dim=action_dim
        self.discount=0.99
        self.entropy_weight=0.5
        self.entropy_eps=1e-6
        self.model_type=model_type

        self.is_central=is_central
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actorNetwork=ActorNetwork(self.s_dim,self.a_dim).double().to(self.device)
        self.cdn_select_net = CDN_Select_NN().double().to(self.device)
        if self.is_central:
            # unify default parameters for tensorflow and pytorch
            self.actorOptim=torch.optim.RMSprop(self.actorNetwork.parameters(),lr=actor_lr,alpha=0.9,eps=1e-10)
            self.actorOptim.zero_grad()
            if model_type<2:
                '''
                model==0 mean original
                model==1 mean critic_td
                model==2 mean only actor
                '''
                self.criticNetwork=CriticNetwork(self.s_dim,self.a_dim).double().to(self.device)
                self.criticOptim=torch.optim.RMSprop(self.criticNetwork.parameters(),lr=critic_lr,alpha=0.9,eps=1e-10)
                self.criticOptim.zero_grad()
        else:
            self.actorNetwork.eval()

        self.loss_function=nn.MSELoss()
    
    def compute_ppo_loss(self,m_probs,a_batch, old_log_probs, td_batch):
        self.clip_ratio = 0.2
        ratio = torch.exp(m_probs.log_prob(a_batch) - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        surrogate1 = ratio * td_batch
        surrogate2 = clipped_ratio * td_batch
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        return actor_loss


    def getNetworkGradient(self,s_batch,a_batch,r_batch,terminal):
        s_batch=torch.cat(s_batch).double().to(self.device)
        a_batch=torch.LongTensor(a_batch).double().to(self.device)
        r_batch=torch.tensor(r_batch).double().to(self.device)
        R_batch=torch.zeros(r_batch.shape).double().to(self.device)

        R_batch[-1] = r_batch[-1]
        for t in reversed(range(r_batch.shape[0]-1)):
            R_batch[t]=r_batch[t] + self.discount*R_batch[t+1]  #accumulated reward (from r_batch of each local agent)

        if self.model_type<2:
            with torch.no_grad():
                v_batch=self.criticNetwork.forward(s_batch).squeeze().double().to(self.device)
            td_batch=R_batch-v_batch # advantage
        else:
            td_batch=R_batch

        probability=self.actorNetwork.forward(s_batch)
        m_probs=Categorical(probability)
        log_probs=m_probs.log_prob(a_batch)
        old_log_probs = log_probs.detach()
        actor_loss = self.compute_ppo_loss(m_probs,a_batch, old_log_probs, td_batch)
        # actor_loss=torch.sum(log_probs*(-td_batch)) # based on the advantage calc.
        entropy_loss=-self.entropy_weight*torch.sum(m_probs.entropy())
        actor_loss=actor_loss+entropy_loss
        actor_loss.backward()  # backpropagation with gradients


        if self.model_type<2:
            if self.model_type==0:
                # original
                critic_loss=self.loss_function(R_batch,self.criticNetwork.forward(s_batch).squeeze())
            else:
                # cricit_td
                v_batch=self.criticNetwork.forward(s_batch[:-1]).squeeze()
                next_v_batch=self.criticNetwork.forward(s_batch[1:]).squeeze().detach()
                critic_loss=self.loss_function(r_batch[:-1]+self.discount*next_v_batch,v_batch)

            critic_loss.backward()
            return critic_loss.item()
        
        return actor_loss.item()

    def actionSelect(self,stateInputs):
        if not self.is_central:
            with torch.no_grad():
                stateInputs = stateInputs.double().to(self.device)
                self.actorNetwork.double().to(self.device)
                probability=self.actorNetwork.forward(stateInputs)
                m=Categorical(probability)
                action=m.sample().item()
                return action

    # data in the source param will be copied to the target_param
    def hardUpdateActorNetwork(self, actor_net_params):
        for target_param, source_param in zip(self.actorNetwork.parameters(),actor_net_params):
            target_param.data.copy_(source_param.data)
            
    def updateNetwork(self):  # finally weights are updated after the .backward() of the actor
        # use the feature of accumulating gradient in pytorch
        if self.is_central:
            self.actorOptim.step()
            self.actorOptim.zero_grad()
            if self.model_type<2:
                self.criticOptim.step()
                self.criticOptim.zero_grad()

    def getActorParam(self):
        return list(self.actorNetwork.parameters())
    
    def getCriticParam(self):
        return list(self.criticNetwork.parameters())
    


if __name__ =='__main__':
    # test maddpg in convid,ok
    SINGLE_S_LEN=19

    AGENT_NUM=1
    BATCH_SIZE=200

    S_INFO=6
    S_LEN=8
    ACTION_DIM=6

    discount=0.9

    obj=A3C(False,0,[S_INFO,S_LEN],ACTION_DIM)

    episode=3000
    for i in range(episode):

        state=torch.randn(AGENT_NUM,S_INFO,S_LEN)
        action=torch.randint(0,5,(AGENT_NUM,),dtype=torch.long)
        reward=torch.randn(AGENT_NUM)
        probability=obj.actionSelect(state)
        print(probability)

  





