# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 14:03:30 2018

@author: 606C
"""
max_num = lambda m, n: m if m>n else n # = def max(m, n):
                                       #       return m if m>n else n
print(max(10,3)) 
'''
lambda是運算式，不是陳述句，你在:之後的也必須是運算式，lambda中也不能有區塊，
這表示一些小的運算任務你可以使用lambda，而較複雜的邏輯你可以使用def來定義。
'''
#結合字典物件與lambda模擬switch的示範：
score = int(input('請輸入分數：'))
level = score // 10
{
    10 : lambda: print('Perfect'),
    9  : lambda: print('A'),
    8  : lambda: print('B'),
    7  : lambda: print('C'),
    6  : lambda: print('D')
}.get(level, lambda: print('E'))()

def c(n):
    return lambda x: x + n
d = c(12)
print(d(5))  # 17
print(d(18)) # 30

