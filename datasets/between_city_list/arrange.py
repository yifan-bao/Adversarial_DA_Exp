import os

from numpy.lib.utils import info

city1 = ['aachen', 'bremen', 'darmstadt', 'erfurt', 'hanover', 'krefeld', 'strasbourg', 'tubingen', 'weimar']
city2 = ['bochum', 'cologne', 'dusseldorf', 'hamburg', 'jena', 'monchengladbach', 'stuttgart', 'ulm', 'zurich']
trainval = 'trainval.txt'
# 打开文件
# 根据'_'进行split
# 找城市-根据城市重新构造名字 -- 每个名字的前10组放到 val里面 这样val就90个内容-不清楚是否少了-以后再修改吧
# 前部分放到train.txt中 后部分放到test.txt中
val1 = 'val1.txt'
val2 = 'val2.txt'
train1 = 'train1.txt'
train2 = 'train2.txt'

f_val1 = open(val1,'w')
f_val2 = open(val2,'w')
f_train1 = open(train1,'w')
f_train2 = open(train2,'w') 

items = [id.strip() for id in open(trainval)]

count_val = 0
old_name = ''

for id in items:
    info_list = id.split('_')
    city = info_list[1]
    if count_val == 0:
        old_name = city
    if old_name == city and count_val < 10:
        # 写val
        if city in city1:
            f_val1.write(id+'\n')
        elif city in city2:
            f_val2.writelines(id+'\n')
        count_val += 1
    elif old_name == city and count_val >= 10:
        # 写train
        if city in city1:
            f_train1.write(id+'\n')
        elif city in city2:
            f_train2.writelines(id+'\n')
        
    elif old_name != city:
        count_val = 0
        #old_name = city

f_val1.close() 
f_val2.close()
f_train1.close()
f_train2.close()