# Lecture 00: Variable and assignment (Solution)

**Exercise:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/00_intro/00_variables_assignment_operator-precedence.ipynb)
**Solution:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/00_intro/00_variables_assignment_operator-precedence_solutions.ipynb)

We begin with assignment to variables and familiar mathematical operations.

## Objectives

- Introduce expressions and basic operators
- Introduce operator precedence
- Understand variables and assignment

<iframe width="560" height="315" src="https://www.youtube.com/embed/eVKPCWBlY8A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/S37dz6osGKo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Aspects of a programming language

### Data objects

Code manipulate data objects. Objects can be classified as two types:
* scalar (cannot be subdivided)
* non-scalar (have internal structure that can be accessed)

Objects have a type that defines the kinds of things programs can do to them. For example, you can multiply 2 numbers, but cannot multiply two strings (words)


```python
5
type(5)
```




    int



We can see that a number is a scalar object, we can perform operations such as multiplying two numbers, which results in another scalar object `int`.


```python
5 * 3
type(5 * 3)
```




    int



There are other scalar types such as: `float`, `str`, `bool` and `None`. `float` represents decimal numbers, `str` is string: a word or a character, `bool` is `True` or `False` and is useful to conditionally evaluate some statements. NoneType: special and has one value,`None`.


```python
type(5.2)
type("Hello")
type(True)
5 == 4
```




    False



### Syntax

Syntax represents the way constituent elements are put together. For example, how words in the English language are put together to form sentences. In python, most often, we need `<object> <operator> <object>` as a syntactically valid statement.


```python
# Syntactically valid
5 * 3
```




    15




```python
# Syntactically invalid
5 "Hi"
```


      File "<ipython-input-5-e973e6e5b6aa>", line 2
        5 "Hi"
          ^
    SyntaxError: invalid syntax



### Static semantics

Static semnatics is when syntactically valid strings have meaning:


```python
5 + "Hi"
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-5-b8aaee428c60> in <module>
    ----> 1 5 + "Hi"
    

    TypeError: unsupported operand type(s) for +: 'int' and 'str'


Although this follows the syntax of `<object> <operator> <object>` it's not semantically valid. In this case, Python was able to say that you are doing an operation that is semantically invalid. Sometimes, Python may do something unintended. For example multiplying an integer by a string is semantically invalid, maybe not what you meant, but would give you the following result.


```python
5 * "Hi"
```

# Evaluating expressions: simple operators

We can use Python like a calculator. Consider the simple expression $3 + 8$. We can evaluate and print this by:


```python
3 + 8
```

Another simple calculation is the gravitational potential $V$ of a body of mass $m$ (point mass) at a distance $r$ from a body of mass $M$, which is given by

$$
V = \frac{G M m}{r}
$$

where $G$ is the *gravitational constant*. A good approximation is $G = 6.674 \times 10^{-11}$ N m$^{2}$ kg$^{-2}$.

For the case $M = 1.65 \times 10^{12}$ kg, $m = 6.1 \times 10^2$ kg and $r = 7.0 \times 10^3$ m, we can compute the  gravitational potential $V$:


```python
6.674e-11*1.65e12*6.1e2/7.0e3
```




    9.59625857142857



The calculation shown above doesn't print out all the digits that has been calculated. It merely does the first 14 decimal places. We can see that by printing 100 decimal places.


```python
print("{:.100f}".format(6.674e-11*1.65e12*6.1e2/7.0e3))
```

We have used 'scientific notation' to input the values. For example, the number $8 \times 10^{-2}$ can be input as `0.08` or `8e-2`. We can easily verify that the two are the same via subtraction:


```python
0.08 - 8e-2
```

A common operation is raising a number to a power. To compute $3^4$:


```python
3**4
```

The remainder is computed using the modulus operator '`%`':


```python
11 % 3
```

To get the quotient we use 'floor division', which uses the symbol '`//`':


```python
11 // 3
```

# Operator precedence

Operator precedence refers to the order in which operations are performed, e.g. multiplication before addition.
In the preceding examples, there was no ambiguity as to the order of the operations. However, there are common cases where order does matter, and there are two points to consider:

- The expression should be evaluated correctly; and
- The expression should be simple enough for someone else reading the code to understand what operation is being
  performed.

It is possible to write code that is correct, but which might be very difficult for someone else (or you) to check.

Most programming languages, including Python, follow the usual mathematical rules for precedence. We explore this through some examples.

Consider the expression $4 \cdot (7 - 2) = 20$. If we are careless,


```python
4*7 - 2
```

In the above, `4*7` is evaluated first, then `2` is subtracted because multiplication (`*`) comes before subtraction (`-`) in terms of precedence. We can control the order of the operation using brackets, just as we would on paper:


```python
4*(7 - 2)
```

A common example where readability is a concern is

$$
\frac{10}{2 \times 50} = 0.1
$$

The code


```python
10/2*50
```

is incorrect. The multiplication and division have the same precedence, so the expression is evaluated 'left-to-right'. The correct result is computed from


```python
10/2/50
```

but this is hard to read and could easily lead to errors in a program. I would recommend using brackets to make the order clear:


```python
10/(2*50)
```

Here is an example that computes $2^{3} \cdot 4 = 32$ which is technically correct but not ideal in terms of readability:


```python
2**3*4
```

Better would be:


```python
(2**3)*4
```

# Variables and assignment

The above code snippets were helpful for doing some arithmetic, but we could easily do the same with a pocket calculator. Also, the snippets are not very helpful if we want to change the value of one of the numbers in an expression, and not very helpful if we wanted to use the value of the expression in a subsequent computation. To improve things, we need *assignment*.

When we compute something, we usually want to store the result so that we can use it in subsequent computations. *Variables* are what we use to store something, e.g.:


```python
c = 10
print(c)
```

Above, the variable `c` is used to 'hold' the value `10`. The function `print` is used to print the value of a variable to the output (more on functions later).

Say we want to compute $c = a + b$, where $a = 2$ and $b = 11$:


```python
a = 2
b = 11
c = a + b
print(c)
```

What is happening above is that the expression on the right-hand side of the assignment operator '`=`' is evaluated and then stored as the variable on the left-hand side. You can think of the variable as a 'handle' for a value.
If we want to change the value of $a$ to $4$ and recompute the sum, we would just replace `a = 2` with `a = 4` and execute the code (try this yourself by running this notebook interactively).

The above looks much like standard algebra. There are however some subtle differences. Take for example:


```python
a = 2
b = 11
a = a + b
print(a)
```

This is not a valid algebraic statement since '`a`' appears on both sides of '`=`', but it is a very common statement in a computer program. What happens is that the expression on the right-hand side is evaluated (the values assigned to `a` and `b` are summed), and the result is assigned to the left-hand side (to the variable `a`). There is a mathematical notation for this type of assignment:

$$
a \leftarrow a +b
$$

which says 'sum $a$ and $b$, and copy the result to $a$'. You will see this notation in some books, especially when looking at *algorithms*.

## Shortcuts

Adding or subtracting variables is such a common operation that most languages provides shortcuts. For addition:


```python
# Long-hand addition
a = 1
a = a + 4
print(a)

# Short-hand addition
a = 1
a += 4
print(a)
```

> In Python, any text following the hash (`#`) symbol is a 'comment'. Comments are not executed by the program;
> they help us document and explain what our programs do. Use lots of comments in your programs.

For subtraction:


```python
# Long-hand subtraction
a = 1
b = 4
a = a - b
print(a)

# Short-hand subtraction
a = 1
b = 4
a -= b
print(a)
```

Analogous assignment operators exist for multiplication and division:


```python
# Long-hand multiplication
a = 10
c = 2
a = c*a
print(a)

# Short-hand multiplication
a = 10
c = 2
a *= c
print(a)

# Long-hand division
a = 1
a = a/4
print(a)

# Short-hand division
a = 1
a /= 4
print(a)
```

## Naming variables

It is good practice to use meaningful variable names in a computer program. Say you used  '`x`' for time, and '`t`' for position, you or someone else will almost certainly make errors at some point.
If you do not use well considered variable names:

1. You're much more likely to make errors.
1. When you come back to your program after some time, you will have trouble recalling and understanding
   what the program does.
1. It will be difficult for others to understand your program - serious programming is almost always a team effort.

Languages have rules for what charcters can be used in variable names. As a rough guide, in Python variable names can use letters and digits, but cannot start with a digit.

Sometimes for readability it is useful to have variable names that are made up of two words. A convention is
to separate the words in the variable name using an underscore '`_`'. For example, a good variable name for storing the number of days would be
```python
num_days = 10
```
Python is a case-sensitive language, e.g. the variables '`A`' and '`a`' are different. Some languages, such as
Fortran, are case-insensitive.

Languages have reserved keywords that cannot be used as variable names as they are used for other purposes. The reserved keywords in Python are:


```python
import keyword
print(keyword.kwlist)
```

    ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']


If you try to assign something to a reserved keyword, you will get an error.

Python 3 supports Unicode, which allows you to use a very wide range of symbols, including Greek characters:


```python
θ = 10
α = 12
β = θ + α
print(β)
```

    22


Greek symbols and other symbols can be input in a Jupyter notebook by typing the `LaTeX` command for the symbol and then pressing the `tab` key, e.g. '`\theta`' followed by pressing the `tab` key.
