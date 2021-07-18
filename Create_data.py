from re import U
from time import time
import cplex
from docplex.mp.model import *
import numpy as np
import pandas as pd
import time

m0=100000
rand_page=500
rand_slot=5
supply_data = pd.DataFrame(np.random.randint(0,2,size=m0), columns=['Gender'])
Gender=np.random.randint(0,2,size=m0)
Location=np.random.randint(0,20,size=m0)
Affinity=np.random.randint(0,10,size=m0)
PageID=np.random.randint(0,rand_page,size=m0)
SlotID=np.random.randint(0,rand_slot,size=m0)
s=np.random.randint(100,1000,size=m0)

supply_data['Location']=Location
supply_data['Affinity']=Affinity
supply_data['PageID']=PageID
supply_data['SlotID']=SlotID
supply_data['s']=s

matrix = np.random.rand(rand_page,5)
f=matrix/matrix.sum(axis=1)[:,None]
page_frequency=pd.DataFrame(range(0,rand_page), columns=['PageID'])
page_frequency['f']=f.tolist()

n=50
demand_data=pd.DataFrame()
Value=np.random.randint(5,100,size=n)
Penanty=Value*0.8
Psi=np.random.randint(1,6,size=n)
demand_data['Value']=Value
demand_data['Penanty']=Penanty
demand_data['Psi']=Psi

supply_data_1=supply_data[['PageID','SlotID','s']]
supply_data_1=supply_data_1.groupby(['PageID','SlotID']).sum()

s_sum=supply_data_1.to_numpy()