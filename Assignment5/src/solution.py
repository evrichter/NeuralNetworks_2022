# Newtwon method implementation in python
 
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

 
def find_critical_point(x0, y0):
    # initialize the function
    x=sym.Symbol('x')
    y=sym.Symbol('y') 
    f=x**2 - y**2 - 3*x*y +4
    # initialize first and second derivatives
    fx = sym.diff(f,x)
    fy = sym.diff(f,y)
    fxx = sym.diff(fx,x)
    fxy = sym.diff(fx,y)
    fyy = sym.diff(fy,y)
    ## the following function transform the function to lambda function
    ## which is easy to deal with in python
    f_lambda=sym.lambdify((x,y),f,'numpy')
    fx_lambda=sym.lambdify((x,y),fx,'numpy')
    fy_lambda=sym.lambdify((x,y),fy,'numpy')
    fxx_lambda=sym.lambdify((x,y),fxx,'numpy')
    fxy_lambda=sym.lambdify((x,y),fxy,'numpy')
    fyy_lambda=sym.lambdify((x,y),fyy,'numpy')
    
    # itialize x and y
    x=x0
    y=y0
    # Calculate the Jacobian and the Hessian
    J = sym.Matrix([fx_lambda(x,y), fy_lambda(x,y)])
    H = sym.Matrix([[fxx_lambda(x,y), fxy_lambda(x,y)],
                    [fxy_lambda(x,y), fyy_lambda(x,y)]])
    # xs and ys are just to save points
    xs = np.empty(0)
    ys= np.empty(0)
    # x_y is the vector carrying the variables
    x_y = sym.Matrix([x,y])
    # lr is the learning rate. We set it to 0.1 to have
    # multiple points to draw in the next excercise 
    # at lr =1 it only takes 1 iteration to reach the minimum
    lr = 0.1
    while J != sym.Matrix([0, 0]):
        # update the variable according to Newton's formula:
        x_y = x_y - lr*H.inv()*J
        x= float(x_y[0])
        y = float(x_y[1])
        xs = np.append(xs,x)
        ys = np.append(ys,y)
        #update the jacobian and hessian according to the new
        # variable values
        J = sym.Matrix([fx_lambda(x,y), fy_lambda(x,y)])
        H = sym.Matrix([[fxx_lambda(x,y), fxy_lambda(x,y)],
                    [fxy_lambda(x,y), fyy_lambda(x,y)]])
        

    return x,y, xs,ys

def plot_function(xs,ys):
    # initialize the function
    x=sym.Symbol('x')
    y=sym.Symbol('y') 
    f=x**2 - y**2 - 3*x*y +4
    f_lambda=sym.lambdify((x,y),f,'numpy')
    # set the range for x and y and format it as mesh
    # for the plot
    x = np.arange(-0.5,0.5,0.01)
    x,y = np.meshgrid(x, x)
    z = f_lambda (x, y)
    # configure the plot to be 3D
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15,7.5))
    # Plot the surface. alpha = 0.4 to make the surface transparent
    # to be able to see the dots
    surf = ax.plot_surface(x, y, z, alpha = 0.4)
    zs = f_lambda(xs,ys)
    ax.scatter(xs,ys,zs, marker="o", color= "red")
    #ax.plot_surface(x,y,f_lambda)
    plt.show()
