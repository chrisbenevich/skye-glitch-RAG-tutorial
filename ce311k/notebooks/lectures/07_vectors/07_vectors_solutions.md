# Lecture 07: Vectors (Solution)

**Exercise:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/07_vectors/07_vectors.ipynb)
**Solution:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/07_vectors/07_vectors_solutions.ipynb)

Working with numbers is central to almost all scientific and engineering computations. 
The topic is so important that there are many dedicated libraries to help implement efficient numerical
computations. There are even languages that are specifically designed for numerical computation, such as Fortran
and MATLAB.

NumPy (http://www.numpy.org/) is the most widely used Python library for numerical computations.  It provides an extensive range of data structures and functions for numerical
computation. In this notebook we will explore just some of the functionality.
You will be seeing NumPy in other courses. NumPy can perform many of the operations that you will learn
during the mathematics courses.

Another library, which largely builds on NumPy and provides additional functionality, is SciPy (https://www.scipy.org/). SciPy provides some  more specialised data structures and functions over NumPy. 
If you are familiar with MATLAB, NumPy and SciPy provide much of what is available in MATLAB.

NumPy is a large and extensive library and this lecture is just a very brief introduction.
To discover how to perform operations with NumPy, your best resources are search engines, such as http://stackoverflow.com/.


## Objectives

- Introduction to 1D arrays (vectors) 
- Manipulating arrays (indexing, slicing, etc)
- Apply elementary numerical operations
- Demonstrate efficiency differences between vectorised and non-vectorised functions

<iframe width="560" height="315" src="https://www.youtube.com/embed/IxQNRp6FbtI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

# Importing the NumPy module

To make NumPy available in our programs, we need to import the module. It has become an informal custom to import NumPy using the shortcut '`np`': 


```python
import numpy as np
```

# Numerical arrays

We have already seen Python 'lists', which hold 'arrays' of data.  We can access the elements of a list using an index because the entries are stored in order. Python lists are very flexible and can hold mixed data types, e.g. combinations of floats and strings, or even lists of lists of lists . . .

The flexibility of Python lists comes at the expense of performance. Many science, engineering and mathematics problems involve very large problems with operations on numbers, and computational speed is important for large problems. To serve this need, we normally use specialised functions and data structures for numerical computation, and in particular for arrays of numbers. Some of the flexibility of lists is traded for performance.

## One-dimensional arrays

A one-dimensional array is a collection of numbers which we can access by index (it preserves order).

### Creating arrays and indexing 

To create a NumPy array of length 5 :


```python
x = np.array([0, 5, 7, 2, 3])

print(x)
print(type(x))
```

    [0 5 7 2 3]
    <class 'numpy.ndarray'>


The default type of a NumPy array is `float`. The type can be checked with


```python
print(x.dtype)
```

    int64


You cannot, for example, add a `string` to a `numpy.ndarray`. All entries in the array have the same type.

We can check the length of an array using `len`, which gives the number of entries in the array:


```python
print(len(x))
```

    5


A better way to check the length is to use `x.shape`, which returns a tuple with the dimensions of the array:


```python
print(x.shape)
```

    (5,)


`shape` tells us the size of the array in *each* direction. We will see two-dimensional arrays shortly (matrices), which have a size in each direction.

We can change the entries of an array using indexing,


```python
print(x)

x[0] = 10
x[3] = -4.3
x[1] = 1

print(x)
```

    [10  1  7 -4  3]
    [10  1  7 -4  3]


Remember that indexing starts from zero!

There are other ways to create arrays, such as an array of 'ones':


```python
y = np.ones(5)
print(y)
```

    [1. 1. 1. 1. 1.]


or an array of zeros


```python
y = np.zeros(5)
print(y)
print(y.dtype)
```

    [0. 0. 0. 0. 0.]
    float64


an array of random values:


```python
y = np.random.rand(6)
print(y)
```

    [0.03738489 0.61038864 0.04065597 0.55285659 0.48534968 0.5884804 ]


or a NumPy array from a Python list:


```python
x = [4.0, 8.0, 9.0, 11.0, -2.0]
y = np.array(x)
print(y)
```

    [ 4.  8.  9. 11. -2.]


To create an empty numpy array


```python
x = np.empty(5)
print(x)
```

    [4.66695232e-310 0.00000000e+000 8.17735622e+141 6.01347002e-154
     8.94213159e+130]


Fill with a scalar value


```python
x.fill(3)
print(x)
```

    [3. 3. 3. 3. 3.]


Two more methods for creating arrays which we will use in later notebooks are:

- `numpy.arange`; and 
- `numpy.linspace`. 

They are particularly useful for plotting functions.
The function `arange` creates an array with equally spaced values. It is similar in some cases to `range`, which we have seen earlier. To create the array `[0 1 2 3 4 5]` using `arange`:


```python
x = np.arange(6)
print(x)
print(type(x))
```

    [0 1 2 3 4 5]
    <class 'numpy.ndarray'>


Note that '6' is not included. We can change the start value, e.g.:


```python
x = np.arange(2, 6, 0.5)
print(x)
```

    [2.  2.5 3.  3.5 4.  4.5 5.  5.5]


The function `linspace` creates an array with prescribed start and end values (both are included), and a prescribed number on values, all equally spaced:


```python
x = np.linspace(0, 100, 6)
print(x)
```

    [  0.  20.  40.  60.  80. 100.]


The `linspace` function is useful for plotting.

### Array arithmetic and functions

NumPy arrays support common arithmetic operations, such as addition of two arrays


```python
x = np.array([1.0, 0.2, 1.2])
y = np.array([2.0, 0.1, 2.1])
print(x)
print(y)

# Sum x and y
z = x + y
print(z)
```

    [1.  0.2 1.2]
    [2.  0.1 2.1]
    [3.  0.3 3.3]


and multiplication of components by a scalar,


```python
z = 10.0*x
print(z)
```

    [10.  2. 12.]


and raising components to a power:


```python
x = np.array([2, 3, 4])
print(x**4)
```

    [ 16  81 256]


We can also apply functions to the components of an array:


```python
# Create an array [0, π/2, π, 3π/2]
x = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
print(x)

# Compute sine of each entry
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = np.sin(x[i])
print(y)
```

    [0.         1.57079633 3.14159265 4.71238898]
    [ 0.0000000e+00  1.0000000e+00  1.2246468e-16 -1.0000000e+00]


We computed the sine of each array entry using `for` loops, but the program becomes longer and harder to read. Additionally, in many cases it will be much slower. 


```python
# Using a vectorized operation
y = np.sin(x)
print(y)
```

    [ 0.0000000e+00  1.0000000e+00  1.2246468e-16 -1.0000000e+00]


The above has computed the sine of each entry in the array `x`. Note that the function `np.sin` is used, and not `math.sin` (which was used in previous notebooks). The reason is that `np.sin` is more general -  it can act on lists/arrays of values rather than on just one value. We will apply functions to arrays in the next notebook to plot functions.

You might see manipulation of arrays without indexing referred to as 'vectorisation'. When possible, vectorisation is a good thing to do. We compare the performance of indexing versus vectorisation below.

### Dot Product of two vectors

The dot product of two vectors $a = [a1, a2, \dots, an]$ and $b = [b1, b2,\dots, bn]$ is defined as:

$ {\displaystyle \mathbf {\color {red}a} \cdot \mathbf {\color {blue}b} =\sum _{i=1}^{n}{\color {red}a}_{i}{\color {blue}b}_{i}={\color {red}a}_{1}{\color {blue}b}_{1}+{\color {red}a}_{2}{\color {blue}b}_{2}+\cdots +{\color {red}a}_{n}{\color {blue}b}_{n}}$

the dot product of vectors $[1, 3, −5]$ and $[4, −2, −1]$ is:

$ {\displaystyle {\begin{aligned}\ [{\color {red}{1,3,-5}}]\cdot [{\color {blue}{4,-2,-1}}]&=({\color {red}1}\times {\color {blue}4})+({\color {red}3}\times {\color {blue}{-2}})+({\color {red}{-5}}\times {\color {blue}{-1}})\\&=4-6+5\\&=3\end{aligned}}} $


```python
# Two vectors of length 4
x = np.array([1., 3, -5])
y = np.array([4, -2, -1.])

# Compute dot-product
dot_product = 0.0
for i in range(len(x)):
    dot_product += x[i]*y[i]

print(dot_product)
```

    3.0



```python
# Using in built dot product
print(np.dot(x, y))
```

    3.0
    CPU times: user 111 µs, sys: 65 µs, total: 176 µs
    Wall time: 157 µs


### Performance example: computing the norm of a long vector

The norm of a vector $x$ is given by: 

$$
\| x \| = \sqrt{\sum_{i=0}^{n-1} x_{i} x_{i}}
$$

where $x_{i}$ is the $i$th entry of $x$. It is the dot product of a vector with itself, 
followed by taking the square root.
To compute the norm, we could loop/iterate over the entries of the vector and sum the square of each entry, and then take the square root of the result.

We will evaluate the norm using two methods for computing the norm of an array of length 10 million to compare their performance. We first create a vector with 10 million random entries, using NumPy:


```python
# Create a NumPy array with 10 million random values
x = np.random.rand(10000000)
print(type(x))
```

    <class 'numpy.ndarray'>


We now time how long it takes to compute the norm of the vector using a custom function. We use the Jupyter 'magic command' [`%time`](Notebook%20tips.ipynb#Simple-timing) to time the operation: 


```python
def compute_norm(x):
    norm = 0.0
    for xi in x:
        norm += xi*xi
    return np.sqrt(norm)

%time norm =compute_norm(x)
print(norm)
```

    CPU times: user 50 µs, sys: 30 µs, total: 80 µs
    Wall time: 107 µs
    5.877381679269498


The time output of interest is '`Wall time`'.

> The details of how `%time` works are not important for this course. We use it as a compact and covenient tool to 
> measure how much time a command takes to execute.

We now perform the same operation with '`numpy.dot`':


```python
%time norm = np.sqrt(np.dot(x, x))
print(norm)
```

    CPU times: user 38 µs, sys: 23 µs, total: 61 µs
    Wall time: 71 µs
    5.877381679269498


You should see that the two approaches give the same result, but the 
NumPy function is more than 100 times faster, and possibly more than 100,000 times faster!

The message is that specialised functions and data structures for numerical computations can be many orders of magnitude faster than your own general implementations. On top of that, the specialised functions are much less 
likely to have bugs!
