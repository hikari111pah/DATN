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

def algorithm_1_full(s,f):
    
    def s_to_omega(s):
        omega=[]
        for i in range(0,len(s)):
            if i==0:
                omega.append(s[len(s)-1])
            else:
                omega.append(s[len(s)-i-1]-s[len(s)-i])
        return omega
    def omega_to_sigma(omega):
        sigma=[]
        for i in range(0,len(omega)):
            sigma.append(omega[i]*(len(omega)-i))
        return sigma
    omega= s_to_omega(s)
    sigma= omega_to_sigma(omega)

    def algorithm_1(omega, sigma,f,V):
        z=[]
        w=[]
        v=[]
        i=j=l=1
        v_t=sum(omega)
        v_a=0
        v_rem=omega[j-1]
        while v_a < v_t :
            if round(abs(v_a-v_t))==0: break
            print('v_a= ',v_a)
            print('v_t= ',v_t)
            print('i= ',i)
            print('j= ',j)
            v_req=(f[i-1]*v_t)
            print('v_req= ',v_req)
            print('v_rem= ',v_rem)
            height=sigma[j-1]/omega[j-1]
            v.append(V-i+1)
    
            if v_req >= v_rem or round(abs(v_req-v_rem))==0 :
                w.append(v_rem)
                j=j+1
                if round(v_rem)==0: j=j-1
                if v_req == v_rem or round(abs(v_req-v_rem))==0:
                    i=i+1
                    v_a = v_a + w[l-1]
                else:
                    f[i-1]=f[i-1] - v_rem/v_t
                    v_a = v_a + w[l-1]
                    v_rem= omega[j-1]
            else:
                w.append(v_req)
                v_a=v_a + w[l-1]
                v_rem= v_rem - w[l-1]
                i=i+1
            z.append(w[l-1]*height)
            l=l+1
        print('z= ',z)
        print('w= ',w)
        print('v= ',v)
        return z, w, v
  
    z, w, v= algorithm_1(omega, sigma,f,V)
    def result_edit(z,w,v):
        z= [round(num) for num in z]
        w= [round(num) for num in w]
        i=0
        while i < len(z):
            if z[i]==0:
                print(i)
                z.pop(i)
                w.pop(i)
                v.pop(i)
            else:
                i=i+1
        return z, w, v
    z,w,v= result_edit(z,w,v)
    return result_edit(z,w,v)

z_last=[]
omega_last=[]
v_last=[]

V=5
for t in range(0,rand_page):
    s_i=np.sort(s_sum[t*5:(t+1)*5], axis=None)[::-1]
    f_i=f[t]
    print('t= ',t)
    print('s_i= ',s_i)
    print('f_i= ',f_i)
    z_i,omega_i,v_i= algorithm_1_full(s_i,f_i)
    print('z_i= ',z_i)
    print('omega_i= ',omega_i)
    print('v_i= ',v_i)
    for a in range(0,len(z_i)):
        z_last.append(z_i[a])
        omega_last.append(omega_i[a])
        v_last.append(v_i[a])

gamma= np.random.randint(2, size=(len(z_last),n))

m=len(z_last)
n=len(gamma[1])
d=len(gamma[1]) #Số đơn hàng
print('z= ', z_last )
print('omega= ', omega_last )
print('v= ', v_last )
print('gamma= ',gamma)
print('Value= ',Value)
print('Penanty= ',Penanty)
print('Psi= ',Psi)
z= z_last
v=v_last
omega=omega_last
d_j=np.zeros(d)
u_j=np.zeros(d)
def Booking_first():
    x_ij=np.zeros((len(z_last), d))
    Booking_1 = Model(name='Booking_1')
    #input: n đơn hàng, m nút
    #       value_j,p_j, psi_j, (d_j)
    #       z_i, omega_i, v_i
    #       gamma(i,j)         
    # create flow variables for each couple of nodes
    # x(i,j) is the flow going out of node i to node j
    x = {i : Booking_1.continuous_var(name='x_{0}_0'.format(i)) for i in range(1,m+1) }
    # each arc comes with a cost. Minimize all costed flows
    Booking_1.maximize(Value[0]*Booking_1.sum(z[i-1]*x[i] for i in range(1,m+1)))

    #tm.print_information()
    for i in range(1,m+1):
        if gamma[i-1][0]==0:
            Booking_1.add_constraint(x[i]==0)
        else:
            Booking_1.add_constraint(x[i]>=0)
    
    for i in range(1,m+1):
        Booking_1.add_constraint(x[i] <= 1)
    for i in range(1,m+1):
        Booking_1.add_constraint(z[i-1]*x[i] <= omega[i-1])
    for i in range(1,m+1):
        if v[i-1] > Psi[0]:
            Booking_1.add_constraint(z[i-1]*x[i] <= Psi[0]*omega[i-1]/v[i-1])
    Result=Booking_1.solve()
    for i in range(1,m+1):
        x_ij[i-1,0]=Result['x_{0}_0'.format(i)]
        print('x_{0}_0= {1}'.format(i,Result['x_{0}_0'.format(i)]))
    return x_ij

x_ij= Booking_first()
if round(sum(z[i]*x_ij[i,0] for i in range(0,m))) < 100000000:
    d_j[0]=np.random.randint(1,round(sum(z[i]*x_ij[i,0] for i in range(0,m))))
else:
    d_j[0]=np.random.randint(1,100000000)
def Booking_all(k):
    Booking_all = Model(name='Booking_all')
    x = {(i,j): Booking_all.continuous_var(name='x_{0}_{1}'.format(i,j)) for i in range(0,m) for j in range(0,k+1)}
    u = {j: Booking_all.continuous_var(name='u_{0}'.format(j)) for j in range(0,k)}
    Booking_all.maximize(Value[k]*Booking_all.sum(z[i]*x[i,k] for i in range(0,m))-Booking_all.sum(Penanty[j]*u[j] for j in range(0,k)))

    for i in range(0,m):
        for j in range(0,k+1):
            if gamma[i][j]==0:
                Booking_all.add_constraint(x[i,j]==0)
            else:
                Booking_all.add_constraint(x[i,j]>=0)
    for j in range(0,k):
        Booking_all.add_constraint(Booking_all.sum(z[i]*x[i,j] for i in range(0,m))+u[j] >= d_j[j])
    for i in range(0,m):
        Booking_all.add_constraint(Booking_all.sum(x[i,j] for j in range(0,k+1)) <= 1)
    for i in range(0,m):
        for j in range(0,k+1):
            Booking_all.add_constraint(z[i]*x[i,j] <= omega[i])
    for i in range(0,m):
        for j in range(0,k+1):
            if v[i-1] > Psi[j-1]:
                Booking_all.add_constraint(z[i]*x[i,j] <= Psi[j]*omega[i]/v[i])
    for j in range(0,k):
        Booking_all.add_constraint(u[j]>=0)
        Booking_all.add_constraint(u[j]<=0.5*d_j[j])
    Result_all=Booking_all.solve()
    for i in range(0,m):
        for j in range(0,k+1):
            x_ij[i,j]=Result_all['x_{0}_{1}'.format(i,j)]
    for j in range(0,k):
        u_j[j]= Result_all['u_{0}'.format(j)]     
    return x_ij,u_j

time_solve=[0]
for k in range(1,d):
    start_time=time.time()
    x_ij,u_j=Booking_all(k)
    end_time=end = time.time()
    time_solve.append(end_time-start_time)
    print('Xét đơn thứ: ',k+1)
    print('Đã tốn ',end_time-start_time)
    if round(sum(z[i]*x_ij[i,k] for i in range(0,m))) < 100000000:
        d_j[k]=np.random.randint(1,round(sum(z[i]*x_ij[i,k] for i in range(0,m))))
    else:
        d_j[k]=np.random.randint(1,100000000)



theta_ij=np.zeros((len(z_last), d))
S_j=np.zeros(100)
for j in range(0,d):
    S_j[j]=sum(z[i]*gamma[i,j] for i in range(0,m))
for i in range(0,m):
    for j in range(0,d):
        theta_ij[i,j]=d_j[j]/S_j[j]

def Allocation(k):
    Allocation = Model(name='Allocation')
    x = {(i,j): Allocation.continuous_var(name='x_{0}_{1}'.format(i,j)) for i in range(0,m) for j in range(0,k+1)}
    u = {j: Allocation.integer_var(name='u_{0}'.format(j)) for j in range(0,k+1)}
    Allocation.minimize(Allocation.sum(Allocation.sum(z[i]*(x[i,j]-theta_ij[i,j])*(x[i,j]-theta_ij[i,j]) for i in range(0,m)) for j in range (0,k+1)) +Allocation.sum(Penanty[j]*u[j] for j in range(0,k+1)))

    for i in range(0,m):
        for j in range(0,k+1):
            if gamma[i][j]==0:
                Allocation.add_constraint(x[i,j]==0)
            else:
                Allocation.add_constraint(x[i,j]>=0)
    for j in range(0,k+1):
        Allocation.add_constraint(Allocation.sum(z[i]*x[i,j] for i in range(0,m))+u[j] >= d_j[j])
    for i in range(0,m):
        Allocation.add_constraint(Allocation.sum(x[i,j] for j in range(0,k+1)) <= 1)
    for i in range(0,m):
        for j in range(0,k+1):
            Allocation.add_constraint(z[i]*x[i,j] <= omega[i])
    for i in range(0,m):
        for j in range(0,k+1):
            if v[i-1] > Psi[j-1]:
                Allocation.add_constraint(z[i]*x[i,j] <= Psi[j]*omega[i]/v[i])
    for j in range(0,k+1):
        Allocation.add_constraint(u[j]>=0)
        Allocation.add_constraint(u[j]<=0.5*d_j[j])
    Result_all=Allocation.solve()
    for i in range(0,m):
        for j in range(0,k+1):
            x_ij[i,j]=Result_all['x_{0}_{1}'.format(i,j)]
    for j in range(0,k+1):
        u_j[j]= Result_all['u_{0}'.format(j)]     
    print('x_ij= ',x_ij)
    print('u_j= ',u_j)
    return x_ij,u_j

start_time=time.time()
x_ij,u_j= Allocation(d-1)
end_time=end = time.time()
time_solve_allocation=end_time-start_time
print("Thời gian phân phối: ",time_solve_allocation)
print('DONE')

#print('DONE')
count_node=0
for i in range(0,m):
    for j in range(0,d):
        if gamma[i,j]==1: 
            count_node= count_node+1
print(count_node)
import matplotlib.pyplot as plt
plt.plot(range(0,d),time_solve)
plt.xlabel('Đơn hàng')
plt.ylabel('Thời gian giải')
plt.show()

plt.plot(range(0,d), d_j, label = "d_j")
plt.plot(range(0,d), u_j, label = "u_j")
plt.xlabel('Đơn hàng')
plt.ylabel('Số lượt xem')

plt.legend()
plt.show()