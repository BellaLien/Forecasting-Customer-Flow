import csv
import pandas as pd
import codecs

df = pd.read_csv('佐麥咖啡//佐麥咖啡20170627_20180331.csv', delimiter=',', usecols=['日期', '時間', '品項'], encoding='utf-8')
DF = df['品項'].tolist()
# f2 = codecs.open('佐麥咖啡//購物籃_佐麥咖啡20170627_20180331.csv', 'w', encoding="utf-8")
# writer = csv.writer(f2)

item2 = []
for item in DF:
    item = str(item)
    tmp = item.replace('[','').replace(']','').replace('\'','')
    item2.append(tmp)

data = pd.DataFrame(item2)
print(data)
# for i in data:  
#     print(data[i])
#     writer.writerow(data[i])

data.to_csv('佐麥咖啡//購物籃_佐麥咖啡20170627_20180331.csv', index = False, encoding='utf-8')
# print(data)
# print(type(item[i]))
# print(DF)