import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import matplotlib.pyplot as plt


#======================================================================================
#Functions needed for uncertainty


def x_data_time_vec(x_data , lag , start_pt , finish_pt ):
    #now need to include data from around the lag time
    
    time_indices = np.arange( -int(lag/2.)  , int(lag/2.))

    
    len_new_array = finish_pt - start_pt
    x_data_time = np.zeros(   (len_new_array  , len(time_indices)) )
    for time_i in range(len(time_indices)):
        time = time_indices[time_i]
        x_data_time[:,time_i] = x_data[start_pt+time : finish_pt+time]
    
    return x_data_time

#define an exponential function to use in fitting
#should create a row of Phi
def fit_func_exp(one_data_pt , num_of_params , rate_min , rate_max):
    rate_v = np.arange( rate_min  , rate_max , (rate_max - rate_min)/float(num_of_params) )
    phi_row = []
    for i in range(num_of_params):
        val = np.exp( rate_v[i]*one_data_pt  )
        phi_row.append(val)
    #we need to flatten so that each time element of the input is
    #given a differnt weight.
    return np.array( phi_row ).flatten()


#define a ploynomial function to use in fitting
#should create a row of Phi
def fit_func_poly(one_data_pt , num_of_params , rate_min , rate_max):
    phi_row = []
    for i in range(num_of_params):
        val = one_data_pt**i
        phi_row.append(val)
    #we need to flatten so that each time element of the input is
    #given a differnt weight.
    return np.array( phi_row ).flatten()




#now define phi itself for exponential
def Phi_exp(input , num_of_params , rate_min , rate_max):
    Phi = []
    for i in range( len(input) ):
        one_data_pt = input[i]
        row = fit_func_exp(one_data_pt , num_of_params , rate_min , rate_max)
        Phi.append(row)
    
    return np.array( Phi )




#now define phi itself for polynomial
def Phi_poly(input , num_of_params , rate_min , rate_max):
    Phi = []
    for i in range( len(input) ):
        one_data_pt = input[i]
        row = fit_func_poly(one_data_pt , num_of_params , rate_min , rate_max)
        Phi.append(row)
    
    return np.array( Phi )




from numpy.linalg import inv
#define a function to find uncertainty from Bayes
def S(Phi , alpha,beta):
    multiply = np.dot( Phi.T  , Phi )
    alpha_trm = alpha*np.identity(len(Phi.T) )
    return inv( alpha_trm + beta*multiply )

def uncertainty_poly( x_new , Phi , alpha , beta  , num_of_params , rate_min , rate_max):
    SN = S(Phi , alpha,beta)
    phi_trm = fit_func_poly(x_new ,num_of_params , rate_min , rate_max)
    multi_one = np.dot( SN , phi_trm )
    multi_two = np.dot( phi_trm.T  , multi_one)
    return 1./beta + multi_two


def uncertainty_poly_mat( Phi_old , Phi_new , alpha , beta  , lag ):
    
    number_ones = 1#len(np.arange( -int(lag/2.)  , int(lag/2.)))
    PHI_OLD = np.insert(Phi_old, np.arange(0 , number_ones), values=1, axis=1)
    PHI_NEW = np.insert(Phi_new, np.arange(0 , number_ones), values=1, axis=1)
    
    SN = S(PHI_OLD , alpha,beta)
    multi_one = np.dot( SN , PHI_NEW.T)
    multi_two = np.dot( PHI_NEW , multi_one)
    vec = np.diag( multi_two )
    beta_trms = (1./beta)*np.ones(len(vec))
    return  (beta_trms + vec)


#======================================================================================
#actual data
print '='*30


#pick a lag
lag = 6
start = 50
finish = 50

x_test = np.arange( -0.5 , 1.5 , 0.0001 )
x_test_1 = np.copy(x_test)


#do the lag:
#x_test_lag = x_data_time_vec(x_test_1, lag , start , len(x_test)-finish )
x_test_lag = x_test[start : len(x_test)-finish ]
x_test_lag = x_test_lag.reshape( (len(x_test_lag) , 1) )



t_test_vec = []
for  i in range(len(x_test_lag)):
    t_test = np.sum(np.cos(2*np.pi*x_test_lag[i]) + np.random.normal(loc=0, scale=0.3**2, size=len(x_test_lag[i])))
    t_test_vec.append(t_test)
t_test_vec = np.array(t_test_vec)



######
#params
poly_order = 13

#now define phi itself for polynomial
PHI = Phi_poly(x_test_lag , poly_order , 1, 2)

#now use skilearn
from sklearn import linear_model
clf = linear_model.BayesianRidge()
clf.fit(PHI, t_test_vec)





#make new data
x_new = np.arange(0.4 , 1.9 , 0.001)
x_new_1 = np.copy(x_new)
#x_new_lag = x_data_time_vec(x_new_1, lag , start , len(x_new)-finish )
x_new_lag = np.copy( x_new )
x_new_lag = x_new_lag[start : len(x_new_lag)-finish ]
x_new_lag = x_new_lag.reshape( (len(x_new_lag) , 1) )

PHI_new = Phi_poly(x_new_lag , poly_order , 1, 2)

t_pred = clf.predict(PHI_new)

#plt.plot(x_new[start:len(x_new)-finish ] , t_pred)
#plt.plot( x_test[start:len(x_test)-finish ] , t_test_vec, 'o')



#unc_vec = []
#for i in range(len(x_new_lag)):
#    unc = uncertainty_poly( x_new_lag[i] , PHI , clf.lambda_ , clf.alpha_  , 20 , 1 , 2)
#    unc_vec.append(unc)
#unc_vec = np.array(unc_vec)


#plt.fill_between(x_new[10:len(x_new)-20 ], t_pred + np.sqrt(unc_vec) , t_pred - np.sqrt(unc_vec), facecolor='#FFB6C1', alpha=1.0, edgecolor='none')

unc_vec_2 = uncertainty_poly_mat( PHI , PHI_new , clf.lambda_ , clf.alpha_  , lag)

#print x_new[start:len(x_new)-finish ][3500]
#print x_new
plt.fill_between(x_new[start:len(x_new)-finish ], t_pred + np.sqrt(unc_vec_2) , t_pred - np.sqrt(unc_vec_2), facecolor='#FFB6C1', alpha=1.0, edgecolor='none')


#plt.plot(unc_vec_2)

plt.show()

exit()














