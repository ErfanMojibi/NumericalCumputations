import Equation as eq
import numpy as np
import sympy as sym
import NonLinearEquation as NLE
#q4
print("q4")
x, conv = eq.gauss_seidel(A=np.matrix([[1, 0, -1], [-0.5, 1, 0.25], [1, -0.5, 1]]), b=np.array([0.2, -1.425, 2], ), x_0=np.array([0.9, 0.8, 0.7]), n=300)
x, conv = eq.gauss_seidel(A=np.matrix([[1, 0, -2], [-0.5, 1, -0.25], [1, -0.5, 1]]), b=np.array([0.2, -1.425, 2], ), x_0=np.array([0.9, 0.8, 0.7]), n=300)

#q5
print("q5")
print("Gauss Seidel 1: ")
x, conv = eq.gauss_seidel(A=np.matrix([[1, -2], [2, 1]]), b=np.array([4, 3]), x_0 = np.array([0, 0]), n=100)
print("Gauss Seidel 2: ")
x, conv = eq.gauss_seidel(A=np.matrix([[1, 1], [1, -2]]), b=np.array([3, 4]), x_0 = np.array([0, 0]), n=100)

#q6
print("q6")
print("Gauss Seidel: ")
x, conv = eq.gauss_seidel(A=np.matrix([[5, 3, 4], [3, 6, 4], [4, 4, 5]]), b=np.array([12, 13, 13], ), x_0=np.array([0, 0, 0]), n=20)
print("eigen values: ",np.linalg.eigvals(conv))
print("Jacobi: ")
x, conv = eq.jacobi(A=np.matrix([[5, 3, 4], [3, 6, 4], [4, 4, 5]]), b=np.array([12, 13, 13], ), x_0=np.array([0, 0, 0]), n=20)
print("eigent values: ", np.linalg.eigvals(conv))

#q7
print("q7")
x, y = sym.symbols('x y')
f = x - 1/2*sym.cos(y)
g = y - 1/2*sym.sin(x)
M = sym.Matrix([f, g])
NLE.newton(f=f, g=g, x=x, y=y, x_0=0, y_0=0, n=100)

#q8
print("q8")
x, y = sym.symbols('x y')
f = x**2 + y**2 - 25
g = x**2 - y**2 - 5
M = sym.Matrix([f, g])
jacob = M.jacobian([x]).T
NLE.newton(f=f, g=g, x=x, y=y, x_0=1, y_0=1, n=100)

#q9
print("q9")
x, y = sym.symbols('x y')
f = 2*x**3 - y**2 - 1
g = x*y**3 - y - 4
NLE.newton(f=f, g=g, x=x, y=y, x_0=4, y_0=5, n=100)
