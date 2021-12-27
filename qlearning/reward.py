import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

#定義障礙的reward
class RewardTable:
    def __init__(self,table):
        self.reward_table=table
    def get_distance(self,start,target):
        '''L2 distance'''
        x=start[0]-target[0]
        y=start[1]-target[1]
        return (x**2+y**2)**0.5
    def caculate_reward(self,x,y):
        '''using if-else rule to construct rewards'''
        total_rewards=0
        distance=100*-(self.get_distance(x,y))#距離越遠，reward越小
        total_rewards+=distance
        if self.reward_table[x]==0:
            total_rewards+= -100#obstcale or not
        if x==y:
            total_rewards+=5000#goal or not
        return total_rewards
