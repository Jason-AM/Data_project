import numpy as np

from numpy import linalg as LA



#======================================================================================
#The design Matrix

def Design_matrix(input , fit_func , *args  ):
    Phi = []

    for i in range( len(input) ):
        one_data_pt = input[i]
        row = fit_func(one_data_pt , *args)
        Phi.append(row)


    return np.array( Phi )



#======================================================================================
# Now the functions required to get the Bayesian Uncertainty

def S(Phi_old , alpha , beta):
    multiply = np.dot( Phi_old.T  , Phi_old )
    alpha_trm = alpha*np.identity(len(Phi_old.T) )
    return LA.inv( alpha_trm + beta*multiply )


def Bayes_unc( Phi_old , Phi_new , alpha , beta  ):
    
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

