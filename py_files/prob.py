import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
from scipy.misc import comb
import numpy as np


ns = np.arange(10,31,5)
p = .09

def bi(n,k,p):
    return comb(n,k) * p**k * (1-p)**(n-k)
plt.close('all')

def vary_panel_size(n):
    ks = np.arange(0,n)
    y = [bi(n,k,p) for k in ks]
    plt.plot(ks,y,label = str(n))

plt.style.use('ggplot')
plt.figure()
for n in ns: vary_panel_size(n)
plt.legend()
plt.xlim(0,8)
plt.title('Binomial Distribution taster bias of {} varying by panel size'.format(p), fontsize = 16)
plt.ylabel('Prob of exactly that many tasters saying !TTB', fontsize = 16)
plt.xlabel('num tasters', fontsize = 16)
plt.tight_layout()
plt.show()


n = 30
b = [bi(n,k,p) for k in np.arange(0,n)]

tasters = 6
print('bias =', p*100, 'percent')

for tasters in range(5,8):
    perc = (1 - sum(b[:tasters]))*100
    print(round(perc,2), 'percent chance of !TTB for', tasters, 'or more tasters of', n)

marie = .06
billy = .09
lindsay = .3
dana = .025
soren = .15

a = [marie,billy,dana,soren]
p = np.mean(a)
a = a+a+a+a+a
n = len(a)
b = [bi(n,k,p) for k in np.arange(0,n)]

tasters = 6
print('bias =' , p*100, 'percent')
for tasters in range(5,8):
    perc = (1 - sum(b[:tasters]))*100
    print(round(perc,2), 'percent chance of !TTB for', tasters, 'or more tasters of', n)

b2 = [[bi(n,k,p) for k in np.arange(0,n)] for p in a]

perc2 = [(1 - sum(b[:tasters]))*100 for b in b2]
