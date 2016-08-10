import numpy as np
from sklearn.metrics import mutual_info_score , normalized_mutual_info_score

def MI_lag(  bins , x , y  , max_lag):
    
    #first need to ensure vectors are made the same length
    #always ensure that y is the wind vector x is the sand

    len_diff = len(y) - len(x)
    print len_diff

    if len_diff >0:
        y = y[:-len_diff]
    elif len_diff < 0:
        x = x[:len_diff]


    mi_vec = []
    for lag in range(max_lag):

        x_info = x[lag:]
        y_info = y[0: len(y) - lag]
        
        c_xy = np.histogram2d(x_info, y_info, bins)[0]
        mi_val = mutual_info_score(None, None, contingency=c_xy)



        mi_vec.append(mi_val)
        print lag , len(mi_vec)

    return np.array(mi_vec)