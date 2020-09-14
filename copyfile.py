import pandas as pd
import os,shutil,string
dir = "/Users/rui/PycharmProjects/test1/Final/mess"

f = open('/Users/rui/desktop/label.csv','r')
res = f.readlines()[1:]
nostateList = []
isstateList = []
for i in res:
    n_str = i.replace(',1', '')
    n_str = n_str.replace(',2','')
    n_str = n_str.replace(',0','')
    n_str = n_str.replace('\n','')
    i_str = i.replace('\n','')
    ilist = i_str.split(',')
    nlist = n_str.split(',')
    nostateList.append(nlist)
    isstateList.append(ilist)
#df without statenumber
df = pd.DataFrame(nostateList)
df.columns = ['labelnumber', 'labelname']
df['labelnumber'] = df['labelnumber'].astype('int')
df = df.set_index('labelnumber')
df = df.sort_index(ascending= True)

#df1 with statenumber
df1 = pd.DataFrame(isstateList)
df1.columns = ['labelnumber', 'labelname','statenumber']
df1['labelnumber'] = df1['labelnumber'].astype('int')
df1 = df1.set_index('labelnumber')
df1 = df1.sort_index(ascending= True)

print(df)
print(df1)

for i in os.listdir(dir):
    oldname = dir +'/'+str(i)
    newname = dir + '/'+ str(df.iloc[int(i), 0])
    shutil.move(oldname, newname)
   # else:
#    print("transfered")



