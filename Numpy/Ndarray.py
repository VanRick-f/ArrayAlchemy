import numpy as np
from numpy.random import rand

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
# x1 = np.random.randint(10,size = 6)        #随机生成6位一维数组
# x2 = np.random.randint(10,size = (3,4))    #随机创建3x4二维数组
# x3 = np.random.randint(10,size = (3,4,5))  # 理解是：坐标轴（z,x,y）
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


'''
np.empty 是 NumPy 库中用于创建未初始化数组的函数
它的核心特点是快速分配内存但不初始化数组元素的值。
因此返回的数组可能包含内存中的随机残留值（“垃圾值”）
下列操作位基础计算数组倒数
'''
# def compute (values):
#     output = np.empty(len(values))
#     for i in range(len(values)):
#         output[i] = 1.0/values[i]
#     return output
# values = np.random.randint(1,10,5)
# result = compute(values)
#
# x = np.arange(9)/np.arange(1,10)   # 一维数组相除，相应元素相除
# x = np.arange(9).reshape(3,3)
# print(2**x)     ## 对每个元素进行2**x运算
# x = np.arange(1,10)
# print("x+5：",x+5)
# print("x-5：",x-5)
# print("x*2：",x*2)
# print("x/2：",x/2)
# print("x//2：",x//2)    #向下整除运算
# print("-x",-x)
# print("x%2",x%2)
# print("x**2",x**2)
# print(abs(x))  #numpy 原始函数名位absolute,可以用来处理复数，返回的是复数的模
# '''三角函数运算'''
# theta = np.linspace(0,np.pi,3)
# print(np.sin(theta))
# print(np.cos(theta))
# print(np.tan(theta))
# print("e^x",np.exp(x))
# print("2^x:",np.exp2(x))
# print("3^x:",np.power(x,3))
# print("ln(x):", np.log(x))
# print("log2(x):", np.log2(x))
# print("log10(x):", np.log10(x))
# print("exp(x)-1:",np.expm1(x))
# print("ln(1+x):", np.log1p(x))
# from scipy import special as sp
# x = np.array([1,5,10])
# print("gamma(x)",sp.gamma(x)) #gamma函数
# print("ln|gamma(x)|:",sp.gammaln(x))
# print("beta(x)",sp.beta(x,2)) # 计算输入值x与固定参数2的 Beta 函数值 B(x,2)
# x = np.array([0,0.3,0.7,1.0])
# print("erf(x)",sp.erf(x))  # erf(x)（误差函数）
# print("erfc(x)",sp.erfc(x)) # erfc(x)（互补误差函数）
# print("erfinv(x)",sp.erfinv(x)) #erfinv(x)（逆误差函数）
# x = np.arange(5)
# y = np.empty(5)
# np.multiply(x, 10,out=y)
# x = np.arange(5)
# y = np.zeros(5)
# np.power(2,x,out = y[::2])
# print(y)
# x = np.arange(1,6)
# print(np.add.reduce(x))  #聚合函数，内部相加
# print(np.multiply.reduce(x))  #聚合函数，把运算结果相乘
# print(np.add.accumulate(x))   #聚合函数，内部相加，保留每一次结果
# print(np.multiply.accumulate(x))   #聚合函数，内部相乘，保留每一次结果
# x = np.arange(1,6)
# print(np.add.outer(x,x)) # 加
# print(np.multiply.outer(x,x)) #乘
# print(np.divide.outer(x,x))  #除
# x = np.random.random((100))
# print(np.sum(x),np.min(x),np.max(x)) # 输出数组和、最小值、最大值,且np的sum方法指导数组维度
# print(x.sum(),x.min(),x.max()) # 这种方式更快，x是nparray类型，所以调用的是np的方法
# x = np.random.randint(1,10,(3,4))
# print(x.sum(0))
# print(x.any())
# print(x.all())
# '''引入pandas操作文件，后续章节会讲到'''
# import pandas as pd
# data = pd.read_csv("president_heights.csv")  #读取csv数据
# heights = np.array(data['height(cm)'])      # 创建数组，为身高列
# print("mean height:",heights.mean())        #平均值
# print("min height:",heights.min())           #最小值
# print("max height:",heights.max())            #最大值
# print("standard deviation:",heights.std())        #标准差
# print("25th percentile:",np.percentile(heights,25))   #25分位
# print("75th percentile:",np.percentile(heights,75))   #75分位
# print("median:",np.median(heights))                      #中位数
# '''生成树状图，后续章节会讲到'''
# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set()
# plt.hist(heights)
# plt.title("height distribution of US presidents")
# plt.xlabel("height(cm)")
# plt.ylabel("number")
# plt.show()

# x = np.ones((1,3))
# y = np.arange(6).reshape(2,3)
# print(x+y)
# '''归一化操作'''
# x = np.random.random((3,3))
# x_mean = x.mean(axis=0)
# print(x_mean)
# x_centered = x - x_mean
# print(x_centered.mean(0))
# import matplotlib.pyplot as plt
# x = np.linspace(0,5,50)
# y = np.linspace(0,5,50)[:,np.newaxis]
# z = np.sin(x) **10 +np.cos(10 + y * x) *np.cos(x)  #最终运算的结果z是50x50
# '''我们用可视化工具展示'''
# plt.imshow(z,origin='lower',extent=[0,50,0,50],cmap='viridis')
# plt.colorbar()
# plt.show()
# x = np.random.randint(1,10,size = (3,4))
# print(x)
# print(np.count_nonzero(x))
# print(np.sum(x<6,0))  #同样可以按轴进行
# print(np.any(x<6))  # 查看是否有，同样可以按轴进行
# print(np.all(x<10))  # 查看是否全部，同样可以按轴进行
# print(np.sum( (x<9) & (x>3) ))  # 逻辑运算符 （&和）、（|或）、（~相当于！）、（^异或）
# x = np.random.randint(0,12,(3,4))
# print(x[x<5])  #直接将结果返回一个数组
# temp1 = (x<5) #为x<5创建一个掩码
# temp2 = ( (x<10) & (x>1) )
# print(x[temp1])
# print(x[temp2])

# x = np.random.randint(100,size = 10)
# print(x)
# print(x[3],x[7],x[2])  # 传统索引方式
# ind  = [3,7,2]
# print(x[ind])
# ind = np.array([[3,7],[4,5]])
# print(x[ind])   # 结果与索引形状一样，与原数组无关，但索引值必须合法
# x = np.arange(12).reshape(3,4)
# print(x)
# row = np.array([0,1,2])
# col = np.array([2,1,3])
# print(x[row,col])    # row:行坐标  col:列坐标
# '''广播机制，row:（3x1) ,col:(0x3)  广播为（3x3） 相应位置对应坐标，返回的也是广播后的形状，与原数组无关'''
# print(x[row[:,np.newaxis],col ])
# x = np.arange(12).reshape(3,4)
# print(x)
# print(x[2,[2,0,1]])  # 花哨索引和普通索引结合，原理还是广播机制
# print(x[1:,[2,0,1]]) # 与切片结合
# mask = np.array([1,0,1,0],dtype=bool)
# row = np.array([0,1,2])
# print(x[row[:,np.newaxis],mask])  # 同样是广播机制进行匹配

# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set()
# mean = [0,0]
# cov = [[1,2],[2,5]]
# x = np.random.multivariate_normal(mean,cov,100) # 二维正态分布  （0，0,1,5，（相关系数ρ）） 2：协方差 = cov（r1r2ρ）相当于根号1根号5ρ
# plt.scatter(x[:,0],x[:,1])
# indices = np.random.choice(x.shape[0],20,replace=False)
# selection = x[indices]
# print(selection)  # 被随机选中的20个数据，对应横纵坐标
# plt.scatter(x[:,0],x[:,1],alpha=0.3)
# plt.scatter(selection[:,0],selection[:,1],facecolors='none',edgecolors='b',s = 200)
# plt.show()

# x = np.arange(10)
# i = np.array([2,1,8,4])
# x[i] = 99   # 使用花哨的索引直接修改数组的值，索引需合法
# print(x)
# x[i]-=10  # 逻辑同上，可以进行任意操作，只要索引合法
# print(x)
# x = np.zeros(10)
# x[[0,0]] = [4,6]  # 不可以对同一个位置同一次操作修改
# print(x)
# i = [2,3,3,4,4,4]
# x[i]+=1    # 这里是相同的，同一个位置只能在第一次修改，后续的修改无效
# print(x)
# np.add.at(x,i,1)  # 这就可以在同一次逻辑中进行累次操作的,x在i的索引上加一
# print(x)
# np.multiply.at(x,i,2) # 逻辑同上，运算为乘法
# print(x)

# import matplotlib.pyplot as plt
# x = np.random.randn(100)
# bins = np.linspace(-5,5,20)
# counts = np.zeros_like(bins)  # 创建一个数组，全零，与bins数组形状类型一致
# i = np.searchsorted(bins, x) # 插入算法，在已排序的bins中寻找插入的位置，使结果依旧有序，对每个x单个处理，前面插入的不会影响后面插入的
# np.add.at(counts, i, 1)  #counts与bins形状一致，为计数器，该函数计算每个区间的点数
# plt.step(bins, counts)
# plt.hist(x,bins,histtype='step')  # 这个直接算出x在bins的分布图
# plt.show()
# def select (x):
#     for i in range(len(x)):
#         swap = i+np.argmin(x[i:])  # 此时x[i:] 是对于i的相对索引
#         (x[i], x[swap]) = (x[swap], x[i])
#     return x
# x = np.array([2,1,4,3,5])
# print(select(x))

# x = np.array([2,3,1,4,10,19,3,4,1,9,5])
# x1 = np.sort(x)   #快速排序
# x2 = np.argsort(x)  # 返回的是原数组经过排序后的索引
# x = np.random.randint(0,10,(4,6))
# print(np.sort(x,0))  #按列排序
# print(np.sort(x,1))  #按行排序
# x = np.array([7,2,3,1,6,5,4])
# x1 = np.partition(x,3)  # 前三个小的在左边，同样可以按轴排序
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# x = np.random.rand(10,2)
# plt.scatter(x[:,0],x[:,1],s = 100)
# plt.show()
# dist_sq = np.sum((x[:,np.newaxis,:] - x[np.newaxis,:,:])**2,2)  ## 使用广播机制，计算所有店之间的距离,按列求和，符合公式
# rsult = np.argsort(dist_sq,axis=1) #按行下标排序，则得到每个点，邻接矩阵，第一列为升序因为，对角线为0，故对角线元素都是最小的
# #同样可以在排序时，只展示最小的k个元素

# name = ['alice','bob','cathy','doug']
# age = [25,45,37,19]
# weight = [55,85.5,68,61.5]
# data = np.zeros(4,dtype={'names':('name','age','weight'),'formats':('U10','i4','f8')})  #长度不超过10Unicode、4int，8float
# data['name'] = name
# data['age'] = age
# data['weight'] = weight
# print(data['name'])  #可根据索引获取值
# print(data[0])
# print(data['name'][-2])  #符合检索，获取倒数第二行的名字
# print(data[data['age']<30 ]['name'])  #符合检索，获取年龄小于30的姓名
#
# tp = np.dtype([('id','i8'),('mat','f8',(3,3))]) #自定义数据类型
# x = np.zeros(1,dtype=tp)  #数据类型为tp
# print(x)
# print(x['mat'][0])
#
# data_rec = data.view(np.recarray)  #结构化数组
# print(data_rec.age)                 #这时被结构化的数组就可以像属性一样获取数据