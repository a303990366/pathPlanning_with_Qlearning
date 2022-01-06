import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from reward import RewardTable
from map_pro import dealWithMap
from qlearning import Qlearning
import imageio
import os

if __name__=="__main__":
    grid_shape=(100,100)
    test=dealWithMap("map.pgm",grid_shape)
    #需要輸入三個參數:pgm檔的路徑、網格大小、起點[需要注意起點的位置是否在不可通過的範圍]
    #環境轉換成矩陣
    matrix=test.matrix_for_discrete.copy()
    matrix[np.where(matrix!=0)]=1

    #define
    observation_size=grid_shape[0]*grid_shape[1]
    action_size=8
    actions=[(-1,1),(-1,0),(-1,-1),(0,1),(0,-1),
                         (1,1),(1,0),(1,-1)]
    #make dict for index-action
    action_dict=dict()
    for index,ac in enumerate(actions):
        action_dict[index]=ac
    #make dict for index-loaction
    loc_list=list(test.discrete_table.keys())
    location_dict=dict()
    for index,xy in enumerate(loc_list):
        location_dict[index]=xy

    gamma=0.8
    episode=50
    start=test.contineous2discrete(test.random_goal())
    goal=test.contineous2discrete(test.random_goal())
    print('start:{0},goal:{1}'.format(start,goal))
    tmp=Qlearning((observation_size,action_size),gamma,location_dict,action_dict,episode,goal,matrix,grid_shape)
    tmp.training()

    tmp.cumulative_reward()

    index_start=list(tmp.location_dict.values()).index(start)
    path=tmp.find_path(index_start)

    
    #----------------------Drawing--------------------------
    ori='./fig'
    need_remove=os.listdir(ori)
    if len(need_remove)>0:
        for filename in need_remove:
            os.remove(os.path.join(ori,filename))
    
    tmp=np.where(matrix==0)
    ox=tmp[0]
    oy=tmp[1]
    ims=[]
    for i in range(1,len(path)):
        #plt.gca()
        plt.plot(ox,oy, ".k")
        plt.plot(start[0], start[1], "og")
        plt.plot(goal[0], goal[1], "xb")
        plt.grid(True)
        plt.axis("equal")
        
        rx=[j[0] for j in path[:i]]
        ry=[j[1] for j in path[:i]]
        plt.plot(rx, ry, "-r")
        figpath='./fig/{}.png'.format(i)
        plt.savefig(figpath)
        ims.append(figpath)
 
    with imageio.get_writer('conse.gif', mode='I') as writer:
        for filename in ims:
            image = imageio.imread(filename)
            writer.append_data(image)
    #for filename in set(ims):
     #   os.remove(filename)

