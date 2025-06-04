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

# data = pd.Series([0.25,0.5,0.75,1.0],index=['a','b','c','d'])
# print('a' in data) #可判断索引存在
# print(data.keys()) #查看索引
# print(list(data.items())) #输出对应关系
# data['e'] = 1.25  #可以根据索引修改，不存在则添加
'''使用显示索引包含尾部，隐式索引不包含尾部'''
# print(data['a':'c'])  #切片
# print(data[0:2])  #隐式切片
# print(data[data>0.3]) #掩码
# print(data[['a','e']])
# '''索引器'''
# data = pd.Series(['a','b','c'],index=[1,3,5])
# '''使用索引器选择显示还是隐式'''
# print(data[1]) #取值操作位隐式索引
# print(data[1:3]) #切片操作为隐式索引
# '''loc都是显示'''
# print(data.loc[1])
# print(data.loc[1:3])
# ''' iloc都是隐式的 '''
# print(data.iloc[1])
# print(data.iloc[1:3])
# area_dict = {'california':423967,'texas':695662,'new york':141297,'florida':170312,'illinois':149995}
# area = pd.Series(area_dict)
# population_dict = {'california':123456,'texas':1234123,'new york':4432156,'florida':2432542,'illinois':3653252}
# pop = pd.Series(population_dict)
# data = pd.DataFrame({'area':area,'popu':pop})  #必须要对每一列数据series结构化才可以
# '''列名必须是字符串&&列名不能与内置方法同名'''
# print(data.area)  #列名可以当属性使用
# print(data.popu)
# print(data.area is data['area'])  #二者相同
# data.iloc[0,1] = 123456  #使用隐式索引修改某单个值
# data['density'] = data['popu']/data['area']  #增加列
# print(data.values) #科学计数法
# print(data.T)  #转置
# print(data.values[0])
# '''使用切片时可以不用索引器，为代码规范一般使用'''
# print(data.iloc[:3,:2]) #规则同切片
# print(data.loc[:'new york',:'popu']) #显示获取
# print(data.loc[data.density>10,:])  #可以使用混合模式
# print(data.loc[data.density>10,['popu','density']])  #可以选择列

# np.random.seed(0)
# ser = pd.Series(np.random.randint(0,10,4))
# print(ser)
# df = pd.DataFrame(np.random.randint(0,10,(3,4)),columns=['A','B','C','D'])
# print(df)
# print(np.exp(ser))  #指数函数
# print(np.sin(df*np.pi /4)) #三角函数，通用函数与numpy中相同

# area_dict = {'california':423967,'texas':695662,'new york':141297}
# area = pd.Series(area_dict)
# population_dict = {'california':123456,'texas':1234123,'illinois':3653252}
# pop = pd.Series(population_dict)
# data = pd.DataFrame({'area':area,'popu':pop})  #必须要对每一列数据series结构化才可以
# print(data.popu / data.area)  #根据索引对齐，没对齐的部分NaN，结果是并集
#
# A = pd.Series([2,4,6],index = [0,1,2])
# B = pd.Series([1,3,5],index = [1,2,3])
# print(A + B)  # 会根据索引对齐
# print(A.add(B,fill_value=0))  #可以设置参数自定义缺失的值
#
# np.random.seed(0)
# A = pd.DataFrame(np.random.randint(0,20,(2,2)),columns=list('AB'))
# B = pd.DataFrame(np.random.randint(0,10,(3,3)),columns=list('ABC'))
# fill = A.stack().mean()  #stack作用是降维,降维后求均值
# print(A.add(B,fill_value=fill))

# np.random.seed(0)
# '''默认是按行运算'''
# A = np.random.randint(10,size = (3,4))
# print(A - A[0])
# df = pd.DataFrame( A, columns=list('QRST') )
# print(df)
# print(df - df.iloc[0])
# '''自定义轴'''
# print(df.subtract(df['R'],0)) #按列
#
# halfrow =df.iloc[0,::2]
# print(df-halfrow)  #结果依旧是索引对齐

# vals1 = np.array([1,None,3,4]) #None表示python类型的缺失值
# # print(vals1.sum())   #如果数据中包含python类型的None，进行求和、平均数时会报错
# vals2 = np.array([1,np.nan,3,4]) #None表示python类型的缺失值
# print(vals2.sum())  #nan类型运算操作结果是nan
# print(np.nansum(vals2))  #忽略缺失值
# print(np.nanmean(vals2)) #忽略缺失值
# '''pandas会把None和Nan看做等价的'''
# vals3 = pd.Series([1,np.nan,3,None])
# print(vals3.sum())  #pandas对象可以对包含None或Nan对象进行运算

# data = pd.Series([1,np.nan,'hello',None])
# '''isnull 和notnull 适用于series和dataframe'''
# print(data.isnull())  #判断是否有缺失值的bool数组
# print(data[data.notnull()])  #掩码的方式作为索引
# print(data.dropna())  #series中剔除缺失值
# df = pd.DataFrame([[1,np.nan,2],
#                   [2,3,5],
#                   [np.nan,4,6]])
# df[3] = np.nan
# print(df)
# print(df.dropna())  #默认情况下，会删除包含缺失值的整行
# print(df.dropna(axis='columns')) #剔除包含缺失值的整列
# print(df.dropna(axis='columns', how='all')) #按列剔除，如果整列都为缺失值则剔除整列
# print(df.dropna(axis='columns', how='any')) #按列剔除，只要该列有缺失值则剔除整列
# print(df.dropna(axis='rows', thresh=3)) #thresh参数指最小非缺失值个数：改行意味最少3个非缺失值

# data = pd.Series([1,np.nan,3,None,4],index = list('abcde'))
# print(data.fillna(0))  #缺失值填补0
# print(data.ffill()) #用缺失值前面的值填补
# print(data.bfill()) #用缺失值前面的值填补
# '''dataframe用法相同，添加轴参数即可'''
# df = pd.DataFrame([[1,np.nan,2],
#                   [2,3,5],
#                   [np.nan,4,6]])
# df[3] = np.nan
# print(df)
# print(df.fillna(0))
# print(df.ffill(axis=0)) #按行，向前找
# print(df.bfill(axis=1)) #按列，向后找

# index = [('california',2000),('california',2010),('new york',2000),('new york',2010),('texas',2000),('texas',2010)]
# population = [33871648,37253956,18976457,19378102,20851820,25145561]
# pop = pd.Series(population,index=index)
# print(pop)
# print(pop.loc[[i for i in pop.index if i[1]==2010]])  #传统方式选择2010的数据
# '''多级索引，当索引是个元祖时'''
# index = pd.MultiIndex.from_tuples(index)  #创建多级索引
# pop = pop.reindex(index)  #将pop索引设置为多级索引
# print(pop[:,2010])  #索引1全选，索引2选择2010

# index = [('california',2000),('california',2010),('new york',2000),('new york',2010),('texas',2000),('texas',2010)]
# population = [33871648,37253956,18976457,19378102,20851820,25145561]
# pop = pd.Series(population,index=index)
# index = pd.MultiIndex.from_tuples(index)  #创建多级索引
# pop = pop.reindex(index)  #将pop索引设置为多级索引
# # pop_df = pop.unstack()  #将多级索引的series变为dataframe
# # pop_df = pop_df.stack() #将dataframe转变为多级索引的series
# pop_df = pd.DataFrame({'total':pop,'under18':[9267089,9284094,4687374,4318033,5906301,6879014]}) #增加一列数据
# print((pop_df['under18']/pop_df['total']).unstack()) #查找低于18的人口占比，并转换为dataframe单层索引
#
# '''创建多级索引 '''
# df = pd.DataFrame(np.random.rand(4,2)
#                   ,index = [['a','a','b','b'],[1,2,1,2]] #index参数设置为二维索引
#                   ,columns=['data1', 'data2'])
# data = {('california',2000):33871648} #将元祖作为键传给pandas
# data = pd.Series(data)
# data = pd.MultiIndex.from_arrays([['a','a','b','b'],[1,2,1,2]])  #根据列表创建
# data = pd.MultiIndex.from_tuples([('a',1),('a',2),('b',1),('b',2)])  #根据元祖创建
# data = pd.MultiIndex.from_product([['a','b'],[1,2] ])  #通过笛卡尔积创建
# pop.index.names = ['state','year']  #为多级索引标签添加name
# print(pop)

# '''多级列索引'''
# index = pd.MultiIndex.from_product([[2013,2014],[1,2]],names=['year','visit'])  #笛卡尔积（name是行索引的）
# columns = pd.MultiIndex.from_product([['bob','guido','sue'],['hr','temp']],names = ['subject','type'])   #笛卡尔积（nams是列索引的）
# print(index)
# print(columns)
# data = np.round(np.random.randn(4,6),1)
# data[:,::2]*=10
# data+=37
# healthy_data = pd.DataFrame(data,index=index,columns=columns)
# print(healthy_data)

# index = [('california',2000),('california',2010),('new york',2000),('new york',2010),('texas',2000),('texas',2010)]
# population = [33871648,37253956,18976457,19378102,20851820,25145561]
# pop = pd.Series(population,index=index)
# index = pd.MultiIndex.from_tuples(index)  #创建多级索引
# pop = pop.reindex(index)  #将pop索引设置为多级索引
# pop.index.names = ['state','population']
# columns = ['population']
# print(pop['california',2000]) #获取单值
# print(pop['california'])  #获取某个层级
# print(pop.loc['california':'new york'])  #切片
# print(pop.loc[:,2010]) #切片
# print(pop[pop>20000000]) #掩码
# print(pop[['california','texas']]) #此处返回多列数据，要用双括号[[]]

# index = pd.MultiIndex.from_product([[2013,2014],[1,2]],names=['year','visit'])  #笛卡尔积（name是行索引的）
# columns = pd.MultiIndex.from_product([['bob','guido','sue'],['hr','temp']],names = ['subject','type'])   #笛卡尔积（nams是列索引的）
# data = np.round(np.random.randn(4,6),1)
# data[:,::2]*=10
# data+=37
# healthy_data = pd.DataFrame(data,index=index,columns=columns)
# print(healthy_data)
# print(healthy_data['guido','hr']) #dataframe索引是取列
# print(healthy_data.iloc[:2,:2])  #切片可以用索引器
# print(healthy_data.loc[:,('bob','hr')])  #可以传递元祖数据
# idx = pd.IndexSlice #简化切片
# print(healthy_data.loc[idx[:,1],idx[:,'hr']])  #第一个表示行索引，第二个是列索引，必须都是多级索引

# index = pd.MultiIndex.from_product([('a','c','b'),(1,2) ])
# data = pd.Series(np.random.rand(6), index=index)
# data.index.names = ['char','int']
# # print(data['a':'c'])  #会报错，因为索引不是按照字母有序的
# data = data.sort_index() #对索引进行排序之后就可以了
# print(data['a':'c'])

# index = [('california',2000),('california',2010),('new york',2000),('new york',2010),('texas',2000),('texas',2010)]
# population = [33871648,37253956,18976457,19378102,20851820,25145561]
# pop = pd.Series(population,index=index)
# index = pd.MultiIndex.from_tuples(index)  #创建多级索引
# pop = pop.reindex(index)  #将pop索引设置为多级索引
# pop.index.names = ['state','year']
# print(pop)
# print(pop.unstack(level=0))  #转换成二维模式，按第0个索引
# print(pop.unstack(level=1)) #转换成二维模式，按第1个索引
# pop_flat = pop.reset_index(name = 'population')  #这是大多原始数据的样子,为数据列添加name，将索引还原为普通列
# print(pop_flat.iloc[0])
# pop = pop_flat.set_index( ['state','year'])  #为源数据添加多级索引【为要设置成索引的列名】

# index = pd.MultiIndex.from_product([[2013,2014],[1,2]],names=['year','visit'])  #笛卡尔积（name是行索引的）
# columns = pd.MultiIndex.from_product([['bob','guido','sue'],['hr','temp']],names = ['subject','type'])   #笛卡尔积（nams是列索引的）
# data = np.round(np.random.randn(4,6),1)
# data[:,::2]*=10
# data+=37
# healthy_data = pd.DataFrame(data,index=index,columns=columns)
# print(healthy_data)
# data_mean = healthy_data.groupby(level = 'year').mean()  #使用gropuby后求平均值
# print(data_mean.T.groupby(level = 'type').mean())  #需要以列求平均值时，先进行转置



# ser1 = pd.Series(['A','B','C'],index = [1,2,3])
# ser2 = pd.Series(['D','E','F'],index = [4,5,6])
# print(pd.concat([ser1,ser2],axis=1))  #水品拼接
# print(pd.concat([ser1,ser2],axis=0))  #垂直拼接

# def make_df(col,ind):
#     data = {c:[str(c) +str(i) for i in ind] for c in col}
#     return pd.DataFrame(data)
# df1 = make_df('AB',[1,2])
# df2 = make_df('CD',[3,4])
# print(df1)
# print(df2)
# print(pd.concat([df1,df2],axis=0))
# print(pd.concat([df1,df2],axis=1))

#def make_df(col,ind):
#     data = {c:[str(c) +str(i) for i in ind] for c in col}
#     return pd.DataFrame(data)
# x = make_df('AB',[0,1])
# y = make_df('AB',[2,3])
# print(pd.concat([x,y])) #拼接的索引会重复
# print(pd.concat([x,y],ignore_index=True)) #添加参数忽略索引
# print(pd.concat([x,y],keys = ['x','y']))  #添加多级索引（包含x0,x1,y0,y1）

def make_df(col,ind):
    data = {c:[str(c) +str(i) for i in ind] for c in col}
    return pd.DataFrame(data)

df1 = make_df('ABC',[1,2])
df2 = make_df('BCD',[3,4])
print(df1)
print(df2)
print(pd.concat([df1,df2],join='inner',axis=0)) #交集合并
print(pd.concat([df1,df2],join='outer',axis=0)) #并集合并
print(pd.concat([df1,df2],axis=0).reindex(columns= df1.columns)) #合并，合并后的列自定义 

