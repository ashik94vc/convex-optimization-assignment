from functions.imported_packages import *

def draw_contour( func, gd_xs, newton_xs, fig, levels=np.arange(5, 1000, 10), x=np.arange(-5, 5.1, 0.05), y=np.arange(-5, 5.1, 0.05)):
    """
    Draws a contour plot of given iterations for a function
    func:       the contour levels will be drawn based on the values of func
    gd_xs:      gradient descent iterates
    newton_xs:  Newton iterates
    fig:        figure index
    levels:     levels of the contour plot
    x:          x coordinates to evaluate func and draw the plot
    y:          y coordinates to evaluate func and draw the plot
    """
    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = func( np.matrix([x[i],y[j]]).T , 0 )

    plt.figure(fig)
    plt.contour( x, y, Z.T, levels, colors='0.75')
    plt.ion()
    plt.show()

    line_gd, = plt.plot( gd_xs[0][0,0], gd_xs[0][1,0], linewidth=2, color='r', marker='o', label='GD' )
    line_newton, = plt.plot( newton_xs[0][0,0], newton_xs[0][1,0], linewidth=2, color='m', marker='o',label='Newton' )

    L = plt.legend(handles=[line_gd,line_newton])
    plt.draw()
    time.sleep(1)

    for i in range( 1, max(len(gd_xs), len(newton_xs)) ):

        line_gd.set_xdata( np.append( line_gd.get_xdata(), gd_xs[ min(i,len(gd_xs)-1) ][0,0] ) )
        line_gd.set_ydata( np.append( line_gd.get_ydata(), gd_xs[ min(i,len(gd_xs)-1) ][1,0] ) )

        line_newton.set_xdata( np.append( line_newton.get_xdata(), newton_xs[ min(i,len(newton_xs)-1) ][0,0] ) )
        line_newton.set_ydata( np.append( line_newton.get_ydata(), newton_xs[ min(i,len(newton_xs)-1) ][1,0] ) )


        L.get_texts()[0].set_text( " GD, %d iterations" % min(i,len(gd_xs)-1) )
        L.get_texts()[1].set_text( " Newton, %d iterations" % min(i,len(newton_xs)-1) )

        plt.draw()
        input("Press Enter to continue...")
