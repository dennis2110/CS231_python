# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:40:38 2018

@author: 606C
"""

print('Art: %5d, Price per Unit: %8.2f' %(453, 59.058))               # Art:   453, Price per Unit:    59.06
print('Art: %5d, Price per Unit: %8.2f' %(12345, 15900.058))          # Art:    12, Price per Unit: 15900.06
print('Art: {1:5d}, Price per Unit: {0:8.2f}'.format(59.058,453))     # Art:   453, Price per Unit:    59.06
print('Art: {a:5d}, Price per Unit: {p:8.2f}'.format(a=453,p=59.058)) # Art:   453, Price per Unit:    59.06

print("{0:<20s}{1:6.2f}".format('123456789:', 6.99)) # Spam & Eggs:           6.99
print("{0:>20s}{1:6.2f}".format('123456789:', 6.99)) #         Spam & Eggs:   6.99

x = 378
print("The value is {:06d}".format(x)) # The value is 000378
x = -378
print("The value is {:06d}".format(x)) # The value is -00378

data = dict(province="Ontario",capital="Toronto")
print("The capital of {province} is {capital}".format(**data)) # The capital of Ontario is Toronto

capital_country = {"United States" : "Washington", 
                   "US" : "Washington", 
                   "Canada" : "Ottawa",
                   "Germany": "Berlin",
                   "France" : "Paris",
                   "England" : "London",
                   "UK" : "London",
                   "Switzerland" : "Bern",
                   "Austria" : "Vienna",
                   "Netherlands" : "Amsterdam"}

print("Countries and their capitals:")
for c in capital_country:
    print("{country}: {capital}".format(country=c, capital=capital_country[c]))
#Countries and their capitals:
#United States: Washington
#US: Washington
#Canada: Ottawa
#Germany: Berlin
#France: Paris
#England: London
#UK: London
#Switzerland: Bern
#Austria: Vienna
#Netherlands: Amsterdam    
for c in capital_country:
    print("{country:<15s}: {capital}".format(country=c, capital=capital_country[c]))
#United States  : Washington
#US             : Washington
#Canada         : Ottawa
#Germany        : Berlin
#France         : Paris
#England        : London
#UK             : London
#Switzerland    : Bern
#Austria        : Vienna
#Netherlands    : Amsterdam
print('element of index 1 is {0[0]}'.format([20, 10, 5])) # element of index 1 is 20
