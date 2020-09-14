import pandas as pd
import os
import shutil

isstateList = []
f = open('/Users/rui/desktop/timeplace.csv','r')
res = f.readlines()[1:]
for i in res:
    i_str = i.replace('\n','')
    ilist = i_str.split(',')
    isstateList.append(ilist)

df = pd.DataFrame(isstateList)
df.columns = ['timestamp', 'time','label', 'score1', 'place1', 'score2', 'place2', 'score3', 'place3','score4','place4','score5','place5']
#print(Flist)
df1 = df[['timestamp','time','label']]
df = df.infer_objects()
df = df.set_index('timestamp')
df = df.sort_index(ascending= True)
print(df)
i = 0
n = 0
print(len(df1))

label1list = df1['label'].tolist()
timelist = df1['time'].tolist()



dic={}
dic1 = dic.fromkeys(label1list).keys()
dlist = list(dic1)
path = '/Users/rui/PycharmProjects/test1/Final/timeset'
path1 = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007'
for item in dlist:

    dirname = path + '/' + str(item)

    if not os.path.exists(dirname):
        os.makedirs(dirname)


while i in range(0,len(df1)):
    i = i + n
    n =  0
    while n in range(0,len(df1)):
        if label1list[i] == label1list[i+n]:
            print(label1list[i] +"   " + label1list[i+n]+"  i = " + str(i)+" n+1 = "+str(n+i))
            print("from "+timelist[i]+" to "+timelist[i+n])
            print("\n")
            if n ==0:
                dir1 = path + '/' + str(label1list[i]) + '/' + str(timelist[i]) + '--to--' + str(timelist[i + n])
                if not os.path.exists(dir1):
                    os.makedirs(dir1)
            else:
                old_dir1 = path + '/' + str(label1list[i]) + '/' + str(timelist[i]) + '--to--' + str(timelist[i + n - 1])
                new_dir1 = path + '/' + str(label1list[i]) + '/' + str(timelist[i]) + '--to--' + str(timelist[i + n])
                if not os.path.exists(new_dir1):
                    os.rename(old_dir1, new_dir1)
            #dir1 = path + '/' + str(i) + '/' + timelist[i]+'--to--'+timelist[i+n]
            file = path1 + '/' + str(timelist[i+n]) + '.jpg'
            file_dir = path + '/' + str(label1list[i]) + '/' + str(timelist[i]) + '--to--' + str(timelist[i + n])
            ful_dir = path + '/' + str(label1list[i]) + '/' + str(timelist[i]) + '--to--' + str(timelist[i + n]) + '/' +str(timelist[i + n])
            if not os.path.exists(ful_dir):
                shutil.copy(file, file_dir)
            n = n+1
        else:
            print(label1list[i] +"   " + str(label1list[i+n])+ "  i = " + str(i) + " n+1 = " + str(n+i))
            print("different!")

            print("\n")

            break



