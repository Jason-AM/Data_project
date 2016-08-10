import numpy as np
import matplotlib.pyplot as plt


#the format is that each set of x an y data thats needs plotting
#is in the form that each row of the input fil is the data that
#needs plotting i.e. x1 = [1,2,3] ; y1 = [4,5,6] and
# x2 = [1,2,4] and y2 = [8 , 6 , 7] then we use as inputs
# data_x = [x1 , x2] ; data_y = [y1 , y2]
#

def multi_plot( data_y , data_x = None  , same_xaxis = None , same_yaxis = None  ):
    
    #number of plots required:
    num_of_plots = len(data_y)
    
    #create frame for the plots
    if same_xaxis == None and same_yaxis == None:
        f , axarr = plt.subplots(num_of_plots, ncols=1, sharex=True, sharey=False)
    else:
        f , axarr = plt.subplots(num_of_plots, ncols=1, sharex=same_xaxis, sharey=same_xaxis)


    for i in range(num_of_plots):

        y_plotting = data_y[i]

        if data_x == None:
            axarr.flat[i].plot( y_plotting)

        else:
            x_plotting = data_x[i]
            axarr.flat[i].plot(x_plotting , y_plotting)

    plt.show()

