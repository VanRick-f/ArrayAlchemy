import pandas as pd
import numpy as np

# data = pd.Series([0.25,0.5,0.75,1.0])  #是一个带索引的一维数组
# print(data.values) #输出值
# print(data.index)  #输出pd.index类型

# data = pd.Series([0.25,0.5,0.75,1.0],index=['a','b','c','d'])  #索引可以自定义
# print(data['a'])  #匹配模式类似于键值对
#
# data  = pd.Series(5,index=['a','b','c','d'])  #值可以是表量，所有值相同，不用重复写
# print(data)
#
# data = pd.Series({2:'a',1:'b',3:'c',4:'d'})  #也可以直接是字典
# print(data[3])
#
# data = pd.Series({2:'a',1:'b',3:'c',4:'d'},index=[3,4])  #只会保留显示的键值对，显示输入后，其他就丢弃
# print(data)

# area_dict = {'california':423967,'texas':695662,'new york':141297,'florida':170312,'illinois':149995}
# area = pd.Series(area_dict)
# population_dict = {'california':123456,'texas':1234123,'new york':4432156,'florida':2432542,'illinois':3653252}
# population = pd.Series(population_dict)
# states = pd.DataFrame({'population':population,'area':area})  #必须要对每一列数据series结构化才可以
# print(states)
# print(states.index)  #索引
# print(states.columns) #标签 （列名）
# print(states['area'])  #也可以访问列
# print(states.loc['texas']) #返回某一行
'''通过字典添加'''
# data = [{'a':i,'b':2**i}
#         for i in range(3)]
# data = pd.DataFrame(data)
# print(data)
# '''通过单列series创建对象'''
# data = pd.DataFrame({'population':population})
# data = pd.DataFrame(population,columns = ['population'])
# print(data)
# '''如果有值丢失，会赋值NaN'''
# data = ({'a':1,'b':2},{'a':3,'c':4})
# data = pd.DataFrame(data)
# print(data)
# '''直接通过二维数组创建，如果不指定索引，默认有序整数'''
# data = pd.DataFrame(np.random.randn(3,2),
#                     columns=['foo', 'bar'],
#                     index=['a', 'b','c'])
# '''可通过结构化的方式创建，总之数据的类型可以自己定义'''
# A = np.zeros(3,dtype = [('A','i8'),('B','f8')])
# A = pd.DataFrame(A)
# ind = pd.Index([2,3,5,7,11])
# '''基本操作和py的数组相同，不同之处在于值不能直接通过索引修改，类似集合'''
# print(ind[::2])
# print(ind.size)
# print(ind.dtype)
# print(ind.shape)
# print(ind.ndim)
# ind1 = pd.Index([1,3,5,7,9])
# ind2 = pd.Index([2,3,5,7,11])
# print(ind1.intersection(ind2))   #a∩b
# print(ind1.union(ind2))          #a∪b
# print(ind1.difference(ind2))     #a-b
# print(ind1.symmetric_difference(ind2))  #对称差集，仅在一个集合中存在的
# print(ind1 & ind2 )  #按位与
# print(ind1 |ind2)    #按位或
# print(ind1 ^ ind2)               #按位异或