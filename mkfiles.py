# coding:utf-8
import sys # モジュール属性 argv を取得するため
import os
import random
import subprocess
import re
#from xml.etree.ElementTree import *
from xml.dom import minidom
argv = sys.argv  # コマンドライン引数を格納したリストの取得


#print (argv)

if (len(argv) != 3):   # 引数が足りない場合は、その旨を表示
    print ('Usage: dir num')
    quit()         # プログラムの終了


xmls=os.listdir(argv[1]+"/Annotations/")

if len(xmls)==0:
    print ("no xml")
    quit()

vallist=None
if int(argv[2])!=0:
    valnum=int( int(argv[2])*0.01*len(xmls))
    print ("xml vals count= "+str(valnum))
    print ("all xmls count= "+str(len(xmls)))
    vallist=random.sample(xmls,valnum)

os.makedirs(argv[1]+"/ImageSets/Main",exist_ok=True)

ftrain = open(argv[1]+"/ImageSets/Main/"+'trainval.txt','w')

fval = open(argv[1]+"/ImageSets/Main/"+'test.txt','w')
#f.write('hoge\n')

#trainvalだけ作成
for i in xmls:
    flag=False

    if vallist==None:
        flag=False
    else:
        for pick in vallist:
            if i==pick:
                #valに書き込み
                flag=True
    sp=i.split(".")
    if flag==True:
        print("write val.txt..."+sp[0])
        fval.writelines(sp[0]+"\n")
    else:
        #print("write trainval.txt..."+sp[0])
        ftrain.writelines(sp[0]+"\n")
ftrain.close()
fval.close()



#annotationに書かれているclass名を抽出

c_list=[]
search_dir=(os.path.join(argv[1],"Annotations"))
file_name_list = os.listdir(search_dir)
for file_name in file_name_list:
  #出てくるとハングるので
  if '.swp' in file_name:
    continue
  xdoc = minidom.parse(os.path.join(search_dir, file_name))
  name = xdoc.getElementsByTagName("name")
  for x in name:
    #print (x.firstChild.data)
    c_list.append(x.firstChild.data)

c_list = list(set(c_list))  
print (c_list)
new_list=[]
#listが20になるようにダミーを登録
for var in range(0, 20):
  if len(c_list) <= var:
    #print (var)
    new_list.append("dammy"+str(var))
  else:
    #print (c_list[var])
    new_list.append(c_list[var])
print (new_list)
    

with open("class.txt", mode='w') as f:
    f.write('\n'.join(new_list))


