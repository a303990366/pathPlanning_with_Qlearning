import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
class dealWithMap:
    def __init__(self,img_path,grid_shape):
        self.img_path=img_path
        self.img=cv2.imread(img_path,-1) #原檔
        self.binary_image=cv2.threshold(self.img.copy(), 220, 255, cv2.THRESH_BINARY)[-1]#黑白圖片
        self.grid_shape=grid_shape#網格輸入
        self.location_x_line=None#網格位置x
        self.location_y_line=None#網格位置Y
        self.grid_img=self.draw_grid()#圖片含有網格      
        self.discrete_table=self.make_check_table()#連續空間-離散空間對照表
        self.bound=self.get_bound()#(y.low,y.high,x.low,x.high)
        self.goal=self.random_goal()#隨機選取goal
        #self.start=start#起始點
        #self.map_with_goal_start=self.get_map_with_goal_start()#建置含有網格、目標、起點的地圖
        self.matrix_for_discrete=self.make_matrix_for_cumulative()#reward_table
    #建立網格
    def draw_grid(self, color=(0, 255, 0), thickness=1):
        '''function:create grid
        variable:
            img:input image
            grid_shape:create N*N grid
            color:just color
            thickness:fill the line
        '''
        img=self.img.copy()
        h, w= img.shape
        rows, cols = self.grid_shape
        dy, dx = h / rows, w / cols
        range_x,range_y=[0],[0]
        # draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)
            range_x.append(x)
        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)
            range_y.append(y)
        range_x.append(w)
        range_y.append(h)
        #self.grid_img=img
        self.location_x_line=range_x
        self.location_y_line=range_y
        return img
    def make_check_table(self):
        '''function:Discrete the space and provide the dict can check
        return:dict,follow the format:(x,y)=[x.lower,x.upper,y.lower,y.upper]
        '''
        tail_x=list(map(lambda x:x-1,self.location_x_line[1:]))
        tail_y=list(map(lambda x:x-1,self.location_y_line[1:]))
        table_for_check=dict()
        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                table_for_check[(row,col)]=[self.location_x_line[row],tail_x[row],self.location_y_line[col],tail_y[col]]
        return table_for_check#左上為(0,0);右下為(N,N)
    def get_bound(self):
        '''獲取可移動空間的邊界'''
        img=self.binary_image.copy()
        range_xy=[]
        map_can_pass=[]
        for i in range(img.shape[1]):
            map_can_pass.append(max(img[:,i]))
        for i in range(len(map_can_pass)):
            if map_can_pass[i]==255:
                range_xy.append(i)
                break
        for i in range(len(map_can_pass)-1,0,-1):
            if map_can_pass[i]==255:
                range_xy.append(i)
                break
        map_can_pass=[]
        for i in range(img.shape[0]):
            map_can_pass.append(max(img[i,:]))
        for i in range(len(map_can_pass)):
            if map_can_pass[i]==255:
                range_xy.append(i)
                break
        for i in range(len(map_can_pass)-1,0,-1):
            if map_can_pass[i]==255:
                range_xy.append(i)
                break
        return range_xy#白色範圍(y.low,y.high,x.low,x.high)
    def random_goal(self):
        '''select random goal from boundary'''
        while True:
            goal_x=random.randint(self.bound[0],self.bound[1])
            goal_y=random.randint(self.bound[-2],self.bound[-1])
            if self.binary_image[goal_x,goal_y]>10:
                return (goal_x,goal_y)
#     def get_map_with_goal_start(self):
#         '''展示一個地圖含有網格、目標、起點'''
#         #img=self.img.copy()
#         img=cv2.imread(self.img_path)
#         img=cv2.circle(img,self.goal,10,(0,0,255),-1)
#         img=cv2.circle(img,self.start,10,(0,0,255),-1)
#         return img
    def contineous2discrete(self,target):
        '''連續座標轉為離散座標'''
        if target[0] in self.location_x_line:
            tmp_x=self.location_x_line.index(target[0])
            
        else:
            for i in range(self.grid_shape[0]):
                if target[0]<=self.location_x_line[i]:
                    tmp_x=i-1
                    break
        if target[1] in self.location_y_line:
            tmp_y=self.location_y_line.index(target[1])
        else:
            for j in range(self.grid_shape[1]):
                if target[1]<=self.location_y_line[j]:
                    tmp_y=j-1
                    break
        turn_target=(tmp_x,tmp_y)
        #print('{}->{}'.format(target,turn_target))
        return turn_target
    def make_matrix_for_cumulative(self):
        '''將像素作為reward放入矩陣'''
        matrix_for_discrete=np.zeros(self.grid_shape)
        for items in list(self.discrete_table.keys()):
            low_x,high_x,low_y,high_y=self.discrete_table[items]
            matrix_for_discrete[items]=np.sum(self.binary_image[low_x:high_x,low_y:high_y])#0=black
        return matrix_for_discrete
