import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from reward import RewardTable
from map_pro import dealWithMap
from qlearning import Qlearning

if __name__=="__main__":
    grid_shape=(50,50)
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
    tmp=np.where(matrix==0)
    ox=tmp[0]
    oy=tmp[1]

    ax = plt.gca()                                 #获取到当前坐标轴信息
    ax.plot(ox,oy, ".k")
    ax.xaxis.set_ticks_position('top')   #将X坐标轴移到上面
    ax.invert_yaxis() 
    ax.plot(start[0], start[1], "og")
    ax.plot(goal[0], goal[1], "xb")
    ax.grid(True)
    ax.axis("equal")
        
    rx=[i[0] for i in path]
    ry=[i[1] for i in path]
    plt.plot(rx, ry, "-r")
    plt.pause(0.01)
    plt.show()
