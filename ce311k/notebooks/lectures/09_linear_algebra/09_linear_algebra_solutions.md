# Lecture 09: Linear Algebra (Solution)

**Exercise:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/09_linear_algebra/09_linear_algebra.ipynb)
**Solution:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/09_linear_algebra/09_linear_algebra_solutions.ipynb)

We will be using NumPy (http://www.numpy.org/) and SciPy (https://www.scipy.org/) to solve system of linear equations


## Objectives

- Solving system of linear equations using Gauss Elimination

<iframe width="560" height="315" src="https://www.youtube.com/embed/j26ImDpLO8g" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/pmQrQ9FFOsc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/dAI3DcuhQBg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


```python
import numpy as np
```

## Solve a linear system of equations

Consider a Matrix $A$, and vectors $x$ and $b$:

$$ 
\begin{bmatrix} 
2 & 4 & 6 \\
4 & 11 & 21 \\
6 & 21 & 52 \\
\end{bmatrix} 
	\begin{bmatrix}
		x_1\\
		x_2\\
		x_3\\
	\end{bmatrix}
	\begin{bmatrix}
		24 \\
		72 \\
		158 \\
	\end{bmatrix}
$$

we use:


```python
A = np.array([[2,4,6], [4,11,21], [6, 21, 52]])
b = np.array([24, 72, 158])
```

Check the length of `A` and `b`


```python
print(A.shape)
print(b.shape)
```

    (3, 3)
    (3,)


The determinant ($\det(A)$) can be computed using functions in the NumPy submodule `linalg`. If the determinant of $A$ is non-zero, then we have a solution.


```python
Adet = np.linalg.det(A)
print("Determinant of A: {}".format(Adet))
```

    Determinant of A: 41.999999999999964


Solve using the inverse of A


```python
Ainv = np.linalg.inv(A)
x = Ainv.dot(b)
print("x = {}".format(x))
```

    x = [2. 2. 2.]


Solution using Gauss Elimination


```python
A = np.array([[2,4,6], [4,11,21], [6, 21, 52]])

b = np.array([24, 72, 158])

x = np.linalg.solve(A, b)
print("x = {}".format(x))
```

    x = [2. 2. 2.]


## Gauss-Seidel iterative approach


```python
import numpy as np
def seidel(A, b, max_iter = 1000):    
    x = np.zeros(b.shape[0])
    for iter in range(max_iter): 
        # Loop through each row
        for i in range(A.shape[0]):         
            # temp variable d to store b[j] 
            d = b[i]
            # Iterate through the columns
            for j in range(A.shape[1]):      
                if(i != j): 
                    d-=A[i][j] * x[j] 
            # updating the value of our solution         
            x[i] = d / A[i][i] 
        
        if np.allclose(np.dot(A, x), b, rtol=1e-8):
            break

    else: # no break
        raise RuntimeError("Insufficient number of iterations")
        
    error = np.dot(A, x) - b
    # returning our updated solution            
    return x, error, iter


A = np.array([[2,4,6], [4,11,21], [6, 21, 52]])

b = np.array([24, 72, 158])

x, error, iter = seidel(A, b)
print("Gauss-Seidel iterations {},\nx: {},\nerror: {}".format(iter, x, error))
```

    Gauss-Seidel iterations 199,
    x: [1.99999825 2.0000013  1.99999968],
    error: [-2.44645356e-07  5.07511544e-07  0.00000000e+00]


## Truss analysis


```python
import math
import numpy as np

A = np.zeros((10, 10))

# Angles
alpha = math.pi/6
beta = math.pi/3
gamma = math.pi/4
delta = math.pi/3

A[0,0] = 1
A[0,4] = np.sin(alpha)

A[1,1] = 1 
A[1,3] = 1 
A[1,4] = np.cos(alpha)

A[2,6] = np.sin(beta) 
A[2,7] = np.sin(gamma)

A[3,3] = -1 
A[3,5] = 1 
A[3,6] = -np.cos(beta) 
A[3,7] = np.cos(gamma)

A[4,2] = 1 
A[4,8] = np.sin(gamma)

A[5,5] = -1 
A[5,8] = -np.cos(delta)

A[6,4] = -np.sin(alpha) 
A[6,6] = -np.sin(beta)

A[7,4] = -np.cos(alpha) 
A[7,6] = np.cos(beta) 
A[7,9] = 1

A[8,7] = -np.sin(gamma) 
A[8,8] = -np.sin(delta)

A[9,7] = -np.cos(gamma) 
A[9,8] = np.cos(delta) 
A[9,9] = -1


# Force
b = np.zeros(10)
b[2] = 100

x = np.linalg.solve(A, b)
print(x)
```

    [ 40.58274196   0.          48.51398804  70.29137098 -81.16548391
      34.30456993  46.86091399  84.02869217 -68.60913985 -93.72182797]


## 3-noded truss

$$ 
\begin{bmatrix} 
0 & 1 & 0 & 1 & \cos(\alpha) & 0 \\
1 & 0 & 0 & 0 & \sin(\alpha) & 0 \\
0 & 0 & 0 & -1 & 0 & -\cos(\beta) \\
0 & 0 & 1 & 0 & 0 & \sin(\beta) \\
0 & 0 & 0 & 0 & -\cos(\alpha) & \cos(\beta) \\
0 & 0 & 0 & 0 & -\sin(\alpha) & -\sin(\beta) \\
\end{bmatrix} 
	\begin{bmatrix}
		V_1\\
		H_1\\
		V_2\\
    F_{12}\\
    F_{13}\\
    F_{23}\\
	\end{bmatrix}=
	\begin{bmatrix}
		0 \\
		0 \\
		0 \\
    0 \\
    -5\\
    10\\
	\end{bmatrix}
$$


```python
import numpy as np
import math

alpha = math.pi/6
beta = math.pi/3

ca = np.cos(alpha)
sa = np.sin(alpha)

cb = np.cos(beta)
sb = np.sin(beta)

# A matrix
A = np.zeros((6, 6))

A[0, 1] = 1
A[0, 3] = 1
A[0, 4] = ca

A[1, 0] = 1
A[1, 4] = sa

A[2, 3] = -1
A[2, 5] = -cb

A[3, 2] = 1
A[3, 5] = sb

A[4, 4] = -ca
A[4, 5] = cb

A[5, 4] = -sa
A[5, 5] = -sb

print(A)

# b
b = np.zeros(6)
b[4] = -5
b[5] = 10
print(b)
```

    [[ 0.         1.         0.         1.         0.8660254  0.       ]
     [ 1.         0.         0.         0.         0.5        0.       ]
     [ 0.         0.         0.        -1.         0.        -0.5      ]
     [ 0.         0.         1.         0.         0.         0.8660254]
     [ 0.         0.         0.         0.        -0.8660254  0.5      ]
     [ 0.         0.         0.         0.        -0.5       -0.8660254]]
    [ 0.  0.  0.  0. -5. 10.]



```python
x = np.linalg.solve(A, b)
print(x)
```

    [  0.33493649  -5.           9.66506351   5.58012702  -0.66987298
     -11.16025404]



```python

```
