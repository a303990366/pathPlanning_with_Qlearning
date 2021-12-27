import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from reward import RewardTable
class Qlearning:    
    def __init__(self,shape,gamma,location_dict,action_dict,episode,goal,matrix,grid_shape):
        self.Q=np.zeros(shape)
        self.grid_shape=grid_shape
        self.shape=shape
        self.matrix=matrix
        self.gamma=gamma
        self.location_dict=location_dict
        self.action_dict=action_dict
        self.episode=episode
        self.goal=goal
        self.reward_list=[]
    def get_index_from_xy(self,xy):
        return list(self.location_dict.values()).index(xy)
    def get_index_from_action(self,action):
        return list(self.action_dict.values()).index(action)    
    def isSafe(self,x,y):
        if x>=self.grid_shape[0] or x<0:
            return False
        if y>=self.grid_shape[1] or y<0:
            return False
        return True
    def rule(self,state,action):
        x=state[0]+action[0]#newaction_x
        y=state[1]+action[1]#newaction_y
        new_state=self.get_index_from_xy((x,y))#將new_state(from state to action)進行index轉換 
        index_state=self.get_index_from_xy(state)#將state進行轉換
        index_action=self.get_index_from_action(action)#將action進行轉換
        all_actions=[(x+act[0],y+act[1]) for act in self.action_dict.values()]#是newstate+所有動作的目標點
        newState2AllAction=[]

        rew=RewardTable(self.matrix)
        for i in all_actions:
            if self.isSafe(i[0],i[1]):
                newState2AllAction.append(rew.caculate_reward(i,self.goal))
        #self.Q[index_state,index_action]=self.R[x,y]+self.gamma*(max(newState2AllAction))
        
        self.Q[index_state,index_action]=rew.caculate_reward((x,y),self.goal)+self.gamma*(max(newState2AllAction))
        return (x,y)#new_state
    def training(self):
        #set a limit
        for time in range(self.episode+1):
            initial=self.location_dict[random.randint(0,self.Q.shape[0]-1)]#randome initial
            option=True
            while option:
                if initial==self.goal:
                    if time%10==0:
                        print('episode:{0},total episode:{1}'.format(time,self.episode))
                    option=False#if we finding the goal ,finish it.
                while True:
                    action=self.action_dict[random.randint(0,7)]#random choose one amnong from all possible actions
                    if self.isSafe(initial[0]+action[0],initial[1]+action[1]):
                        break
                next_state=self.rule(initial,action) #!!!
                initial=next_state
            self.reward_list.append(np.sum(self.Q))
            
    def cumulative_reward(self):
        plt.plot(self.reward_list)
        plt.xlabel("Epoches")
        plt.ylabel("Cumulative reward")
        plt.show()
    def find_path(self,start):
        path_list=[self.location_dict[start]]
        while True:
            if self.location_dict[start]==self.goal:
                break
            for action in np.argsort(self.Q[start,:])[::-1]:
                x=self.location_dict[start][0]+self.action_dict[action][0]
                y=self.location_dict[start][1]+self.action_dict[action][1]
                if self.isSafe(x,y)==True:
                    if (x,y) not in path_list:
                        break
            start=(x,y)
            path_list.append(start)
            start=list(self.location_dict.values()).index(start)
        return path_list
