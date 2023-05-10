import numpy as np
import autograd.numpy
from autograd import jacobian
import sympy as sym

sym.init_printing() 

def newton(f, g, x, y, x_0, y_0, n):
    M = sym.Matrix([f, g])
    jacob = M.jacobian([x, y])
    delta_y = M.jacobian([x]).col_insert(1, sym.Matrix([f,g]))
    delta_x = sym.Matrix([f,g]).col_insert(1, M.jacobian([y]))
    
    det_j = sym.lambdify([x,y], jacob.det(), 'numpy')
    det_x = sym.lambdify([x,y], delta_x.det(), 'numpy')
    det_y = sym.lambdify([x,y], delta_y.det(), 'numpy')
    
    counter = 0
    x_i = x_0
    y_i = y_0
    
    while counter < n:
        x_iplus = x_i - det_x(x_i, y_i)/det_j(x_i, y_i)
        y_iplus = y_i - det_y(x_i, y_i)/det_j(x_i, y_i)
        print(f"{counter + 1}: (x,y) = ({x_iplus}, {y_iplus})")
        x_i = x_iplus
        y_i = y_iplus
        counter += 1
    
    return x_i, y_i