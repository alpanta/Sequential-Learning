import numpy as np
import matplotlib.pyplot as plt
import sys

def f(x):
    global a
    return 0.5*(np.tanh(a*(x-0.45))+1)

def actor(I):

    global a   # This 'a' is the 'a' in tanh(ax) in the activation function.
    global n_loop # The number of the loops for different inputs. Here it is 3.
    global W_r
    global initial # The initial conditions of the premotor and motor units in.
    global pp # To be used in actor. These are the value of cortex and striatum.
    global rr # In premotor loops, which will be used for changing W_r.

    iter=100
    a=3
    wd1=0.5
    wd2=0.5
    W_diff=np.array([[wd1,wd2,wd2],[wd2,wd1,wd2],[wd2,wd2,wd1]])
    W_r_motor=0.5*np.ones((n_loop,1))

    p_motor=np.random.rand(3,110)
    m_motor=np.random.rand(3,110)
    r_motor=np.random.rand(3,110)
    n_motor=np.random.rand(3,110)
    d_motor=np.random.rand(3,110)

    p=np.random.rand(3,110)
    m=np.random.rand(3,110)
    r=np.random.rand(3,110)
    n=np.random.rand(3,110)
    d=np.random.rand(3,110)

    # The order (pmrnd)
    p_motor[:,[0]]=initial[:,[0]]
    m_motor[:,[0]]=initial[:,[1]]
    r_motor[:,[0]]=initial[:,[2]]
    n_motor[:,[0]]=initial[:,[3]]
    d_motor[:,[0]]=initial[:,[4]]

    p[:,[0]]=initial[:,[5]]
    m[:,[0]]=initial[:,[6]]
    r[:,[0]]=initial[:,[7]]
    n[:,[0]]=initial[:,[8]]
    d[:,[0]]=initial[:,[9]]

    noise=0.01*np.random.rand(n_loop,1)

    for k in range(1,iter+1):

        p[:,[k]]=f(0.5*p[:,[k-1]]+m[:,[k-1]]+I)
        m[:,[k]]=f(p[:,[k-1]]-d[:,[k-1]])
        r[:,[k]]=W_r*f(p[:,[k-1]])
        n[:,[k]]=f(p[:,[k-1]])
        d[:,[k]]=f(-r[:,[k-1]]+np.matmul(W_diff,n[:,[k-1]]))

        p_motor[:,[k]]=f(0.5*p_motor[:,[k-1]]+m_motor[:,[k-1]]+0.03*p[:,[k-1]]+noise)
        m_motor[:,[k]]=f(p_motor[:,[k-1]]-d_motor[:,[k-1]])
        r_motor[:,[k]]=W_r_motor*f(p_motor[:,[k-1]])
        n_motor[:,[k]]=f(p_motor[:,[k-1]])
        d_motor[:,[k]]=f(-r_motor[:,[k-1]]+np.matmul(W_diff,n_motor[:,[k-1]]))

    pp=p[:,[k]]
    rr=r[:,[k]]

    return np.rint(p_motor[:,[k]])

'''
The reinforcement learning mechanism is modeled. The classic actor critic
model is changed. Two C-BG-TH system is used for the actor part. These are
the premotor and motor loops. The premotor loops has choose-2 type fixed
points when w_r is large. So increasing w_r will cause the premotor loop
to select two actions at the same time. The motor loops, whose input comes
from the output of the premotor loops, will choose then one of these two
actions by the help of noise.
The critic part is not modeled as neural loops. But this is also possible.
The critic part evaluates the values and then generates the TD signal,
which affects the w_r and thus changes the bifurcation characteristic of
the premotor loops namely, the graph of p-I. When w_r is large then the
graph of p-I in the p1=p2 and I1=I2 subspace shows that the small fixed
point bifurcate to a large one as I increases.
'''
a=3
n_loop=3
W_r=0.5*np.ones((n_loop,1))
pp=np.random.rand(3,1)
rr=np.random.rand(3,1)

input('\nPress any key to START!\n')

mu=0.95  # The discount factor in reinforcement learning
mu_v=0.1 # Learning rate for value function
mu_c=0.1 # Learning rate for association matrix
mu_r=0.2 # Learning rate for W_r weights in Basal Ganglia

'''
THE TRAINING PHASE HAS THREE STAGES. FIRST C IS GIVEN AND 3 IS DESIRED AND A
REWARD IS GIVEN IF THE SUBJECT DOES THE TRUE ACTION. THIS IS CALLED THE C3
STAGE. THEN B IS GIVEN AND 2 IS DESIRED. IF THE SUBJECT DOES THE TRUE ACTION
THEN NO REWARD BUT C IS GIVEN AND 3 IS DESIRED. A reward is given IF THE SUBJECT
DOES THE TRUE ACTION. THIS IS CALLED B2C3 STAGE. AND THEN COMES THE A1B2C3
STAGE.
'''

okey=0

A=np.array([1,0,0]).reshape(3,1)
B=np.array([0,1,0]).reshape(3,1) # Stimulants
C=np.array([0,0,1]).reshape(3,1)

Y1=np.array([1,0,0]).reshape(3,1)
Y2=np.array([0,1,0]).reshape(3,1) # Desired outputs
Y3=np.array([0,0,1]).reshape(3,1)

Y=np.concatenate((Y1,Y2,Y3), axis=1)
X=np.concatenate((A,B,C,np.zeros(3).reshape(3,1)), axis=1)

testtype=np.array([3,2,1]) # Correspond to C3, B2C3, A1B23
W_c=0.1+0.5*np.random.rand(n_loop,n_loop) # Association matrix
W_v=0.01*np.random.rand(1,3) # Linear value function
base=0.2 # Base value for the value function

V=list()
reward=list()
delta=list()                                                                    #
deltalist=list()
I=np.random.rand(3,1500)    # Definitions
II=np.random.rand(3,1500)
action=np.random.rand(3,1500)
W_r_v=np.random.rand(3,1500)

V.insert(0,np.matmul(W_v,C)) # Output of the value function
numberoftest=10000
kk=0

for tt in range(1,4):
    for k in range(1,numberoftest+1):
        for t_type in range(testtype[tt-1],4):


            kk+=1
            initial=0.1*np.random.rand(n_loop,10)
            I[:,[kk-1]]=X[:,[t_type-1]]
            V.insert(kk-1,np.matmul((W_v+base),I[:,[kk-1]]))
            II[:,[kk-1]]=np.matmul(W_c,I[:,[kk-1]])
            action[:,[kk-1]]=actor(II[:,[kk-1]]) # Actor has two parts;
                                                 # Premotor and motor loops.

            if np.array_equal(action[:,[kk-1]],Y[:,[t_type-1]]):
                okey+=1
                if np.array_equal(action[:,[kk-1]],Y[:,[2]]):
                    reward.insert(kk-1,1)
                else:
                    reward.insert(kk-1,0)

                I[:,[kk]]=X[:,[t_type]]
                V.insert(kk,np.matmul((W_v+base),I[:,[kk]]))
                delta.insert(kk-1,reward[kk-1]+mu*V[kk]-V[kk-1])
                deltalist.append((reward[kk-1]+mu*V[kk]-V[kk-1]).item())
                W_v=W_v+mu_v*delta[kk-1]*(I[:,[kk-1]].T)

                for ii in range(0,3):
                    if W_v[:,[ii]]<0:
                        W_v[:,[ii]]=0

                V.insert(kk,np.matmul((W_v+base),I[:,[kk]]))
                W_r=W_r+mu_r*delta[kk-1]*f(pp)*rr
                if np.amax(W_r)>1:
                    W_r=W_r/np.amax(W_r)

                W_c=W_c+mu_c*delta[kk-1]*np.matmul(action[:,[kk-1]],(I[:,[kk-1]].T))

                if np.array_equal(I[:,[kk-1]],X[:,[0]]):
                    print('The letter -A-.')
                elif np.array_equal(I[:,[kk-1]],X[:,[1]]):
                    print('The letter -B-.')
                elif np.array_equal(I[:,[kk-1]],X[:,[2]]):
                    print('The letter -C-.')

                temp1=np.array([[reward[kk-1]],[okey],[0]])
                temp2=np.concatenate((action[:,[kk-1]],W_r,W_c,W_v.T,temp1),axis=1)
                print('\nANS =')
                np.savetxt(sys.stdout, temp2, fmt="%.3f")

            else:
                okey=0
                reward.insert(kk-1,0)
                I[:,[kk]]=X[:,[3]]
                V.insert(kk,np.matmul((W_v+base),I[:,[kk]]))
                delta.insert(kk-1,reward[kk-1]+mu*V[kk]-V[kk-1])
                deltalist.append((reward[kk-1]+mu*V[kk]-V[kk-1]).item())
                W_v=W_v+mu_v*delta[kk-1]*(I[:,[kk-1]].T)

                for ii in range(0,3):
                    if W_v[:,[ii]]<0:
                        W_v[:,[ii]]=0

                V.insert(kk,np.matmul((W_v+base),I[:,[kk]]))
                W_r=W_r+mu_r*delta[kk-1]*f(pp)*rr
                if np.amax(W_r)>1:
                    W_r=W_r/np.amax(W_r)

                W_c=W_c+mu_c*delta[kk-1]*np.matmul(action[:,[kk-1]],(I[:,[kk-1]].T))

                if np.array_equal(I[:,[kk-1]],X[:,[0]]):
                    print('The letter -A-.')
                elif np.array_equal(I[:,[kk-1]],X[:,[1]]):
                    print('The letter -B-.')
                elif np.array_equal(I[:,[kk-1]],X[:,[2]]):
                    print('The letter -C-.')

                temp3=np.array([[reward[kk-1]],[okey],[0]])
                temp4=np.concatenate((action[:,[kk-1]],W_r,W_c,W_v.T,temp3),axis=1)
                print('\nANS =')
                np.savetxt(sys.stdout, temp4, fmt="%.3f")

                break

        print('OK = {0}'.format(okey))

        if okey>40:
            print('\nNumber of success in this stage is {0}.'.format(okey))
            okey=0
            if tt==1:
                print('C-3 stage is done.')
                input('To CONTINUE press a button!\n')
            elif tt==2:
                print('B-2 C-3 stage is done.')
                input('To CONTINUE press a button!\n')
            else:
                print('A-1 B-2 C-3 stage is done.')
                input('To EXIT press a button!')

            break

        W_r_v[:,[kk-1]]=W_r[:]

        print('Trial-{0}\n'.format(kk))

# Plots
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4)
ax1.plot(W_r_v[0,:kk], 'm')
ax2.plot(W_r_v[1,:kk], 'c')
ax3.plot(W_r_v[2,:kk], 'r')
ax4.plot(deltalist, 'b')
ax4.title.set_text('Reinforcement Signal-Delta')
fig.tight_layout()
plt.show()
