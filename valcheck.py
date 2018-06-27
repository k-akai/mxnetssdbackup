#!/usr/bin/python
# coding: UTF-8
import re
f = open('tekito.txt')
lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
f.close()
# lines2: リスト。要素は1行の文字列データ
epochflag=None
epochlist=[]
vallist=[]
name="crack"
for line in lines2:
    if line.find("epoch") >-1:
        epochlist.append(line)
    elif line.find(name):

        num = re.search('[0-9]{1}.[0-9]*',line)
        if num:
            print num.group()
            vallist.append(float(num))

