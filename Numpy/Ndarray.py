import numpy as np


a1 = np.array([1,5,3,7,9])                #创建数组,数据类型相同
a2 = np.array([1.5,5,3,7,9])              #若数据类型不同，则会向上转型，此时转为浮点型
a3 = np.array([range(i,i+3) for i in [2,4,6]]) # 二维数组
a4 = np.zeros(10,dtype=int)  # 创建一个10位整形数组，全为0
a5 = np.ones((3,5),dtype=float) # 创建一个3x5二维浮点型数组，全为1
a6 = np.full(10,3.14,dtype=float)  #创建一个10位浮点型数组，全为3.14
a7 = np.linspace(0,1,5) # 创建一个数组，0开始，1结束，均匀分5份
a8 = np.random.random((3,3)) # 随机创建一个3x3数组，在0-1内均匀分布
a9 = np.random.normal(0,1,(3,3)) # 随机创建一个3x3数组，满足N~(0,1)
a10 = np.random.randint(0,10,(3,3)) #随机创建一个3x3数组，满足0-10
a11 = np.eye(3)  # 三阶单位矩阵

np.random.seed(0) # 设置种子值：0
x1 = np.random.randint(10,size = 6)        #随机生成6位一维数组
x2 = np.random.randint(10,size = (3,4))    #随机创建3x4二维数组
x3 = np.random.randint(10,size = (3,4,5))  # 理解是：坐标轴（z,x,y）
# print(x2[0,1])     # 可以使用x2[0,1] 也可以用x2[0][1]

# print(a1[0:5:1])   # 和python列表一样[start:stop:step]

# print(x2[:2,:3])    # 中间用逗号分隔，两行，三列
# print(x2[::-1,::-1 ])   # 输出所有行列，行逆序，列逆序，步长都为1
# print(x2[::,1])    # 获取第[1]列全部元素
# print(x2[1,:])   # 获取行时等价于 x2[1]

'''此时x2[0,0]的数据会变成100'''
# x2_sub = x2[:2,:2]
# x2_sub[0,0] = 100
# print(x2)
'''使用copy方法明确使用切片的副本，不会修改原数据'''
# x2_sub = x2[:2,:2].copy()
# x2_sub[0,0] = 100

# grid = np.arange(1,10).reshape(3,3)    # 将一维数组改为3x3的矩阵
# x = x1.reshape(6,1)  # 将x1 变换为列向量
# print(x1[:,np.newaxis])  # newaxis为增加维度的关键字，此处为增加列
# print(x1[np.newaxis,:])  # 此处为插入行维度

# y1 = np.array([1,2,3])
# y_1 = np.concatenate([y1,y1])   #拼接两个数组
# y_2 = np.concatenate([y1,y1,y1,y1,y1,y1]) #拼接多个数组
# '''当前维度'''
# z1 = np.concatenate([x2,x2],axis=0)  #拼接二维数组，以第一个轴拼接(行)
# z2 = np.concatenate([x2,x2],axis=2)  #拼接二维数组，以第二个轴拼接(列)
# '''当前维度拼接，是concatenate的便捷封装'''
# s1 = np.vstack([x2,x2])   # 固定轴，垂直拼接
# s2 = np.hstack([x2,x2])  # 固定轴，水平拼接
#
# '''升维拼接'''
# '''
#    result[0,i, j] = A[i, j]
#    result[1,i, j] = B[i, j]
# '''
# z2 = np.stack([x2,x2],0)  # 按层拼接
# '''
#    result[i, 0, j] = A[i, j]
#    result[i, 1, j] = B[i, j]
# '''
# z2 = np.stack([x2,x2],1)  # 按行拼接
# '''result[i, j, 0] = A[i, j]
#    result[i, j, 1] = B[i, j]
# '''
# z2 = np.stack([x2,x2],2)  # 按列拼接
# s3 = np.dstack([x2,x2],)  # 根据深度拼接 等价于axis:2

# s1,s2,s3 = np.split(x1,[4,5])   # 分裂x1数组，在4,5两个位置之前截取
# s1,s2 = np.vsplit(x2,[2])      #垂直分裂，逻辑同上
# s1,s2 = np.hsplit(x2,[1])      #水平分裂，逻辑同上
# s1,s2 = np.vsplit(x3,[1])      #按层切
# s1,s2 = np.hsplit(x3,[1])      #按行切
# s1,s2 = np.dsplit(x3,[1])      #按列切



