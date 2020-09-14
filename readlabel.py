import pandas as pd

f = open('categories_places365.txt','r')
res = f.readlines() #读取 以列表形式储存
newList = []
newlist = []
stateNumber = 1
for i in res:
    n_str = i.replace('/a/', '')
    n_str = n_str.replace('/b/', '')
    n_str = n_str.replace('/c/', '')
    n_str = n_str.replace('/d/', '')
    n_str = n_str.replace('/e/', '')
    n_str = n_str.replace('/f/', '')
    n_str = n_str.replace('/g/', '')
    n_str = n_str.replace('/h/', '')
    n_str = n_str.replace('/i/', '')
    n_str = n_str.replace('/j/', '')
    n_str = n_str.replace('/k/', '')
    n_str = n_str.replace('/l/', '')
    n_str = n_str.replace('/m/', '')
    n_str = n_str.replace('/n/', '')
    n_str = n_str.replace('/o/', '')
    n_str = n_str.replace('/p/', '')
    n_str = n_str.replace('/q/', '')
    n_str = n_str.replace('/r/', '')
    n_str = n_str.replace('/s/', '')
    n_str = n_str.replace('/t/', '')
    n_str = n_str.replace('/u/', '')
    n_str = n_str.replace('/v/', '')
    n_str = n_str.replace('/w/', '')
    n_str = n_str.replace('/x/', '')
    n_str = n_str.replace('/y/', '')
    n_str = n_str.replace('/z/', '')
    n_str = n_str.replace(' ', ',')
    n_str = n_str.replace('\n', '')
    nlist = n_str.split(',')
    nlist.append(int(stateNumber))
    newList.append(nlist)

df = pd.DataFrame(newList)
df.columns = ['labelname', 'labelnumber', 'statenumber']
print(df)

df['labelnumber'] = df['labelnumber'].astype('int')
df = df.set_index('labelnumber')
df = df.sort_index(ascending= True)
print (df)
save_path = "/Users/rui/desktop/label.csv"
df.to_csv(save_path,sep=',',index=True,header=True)


