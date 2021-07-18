import Create_data
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