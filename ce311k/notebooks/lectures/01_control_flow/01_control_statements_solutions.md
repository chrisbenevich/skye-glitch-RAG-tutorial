# Lecture 01a: Control statements (Solution)

**Exercise:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/01_control_flow/01_control_statements.ipynb)
**Solution:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/01_control_flow/01_control_statements_solutions.ipynb)

Control statements allow a program to change what it does depending on input or other data.
Typical flows in a computer program involve structures like:

- if 'X' do task 'A', else if 'Y' do task 'B'
- perform the task 'A' 'N' times
- perform the task 'B' until 'X' is true

These flows are implemented using what are called 'control statements'. They are also known as branching - the path a program follows depends on the input data. Control statements are a major part of all non-trivial computer programs.

<iframe width="560" height="315" src="https://www.youtube.com/embed/6sJBEGm9rQk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/MH1XO_gpWhY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/VvSwrhK1HPs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Objectives

- Introduce Boolean types
- Introduce comparison operators
- Learn to use control statements


## Example of a control statement in pseudo code

An electric window opener, attached to a rain sensor and a temperature 
gauge, might be controlled by the following program:

    if raining:  # If raining, close the window
        close_window()
    else if temperature > 80:  # If the temperature is over 26 deg, open window
        open_window()
    else if temperature < 66:  # If the temperature is below 19 deg, close window
        close_window()
    else:  # Otherwise, do nothing and leave window as it is
        pass

It is easy to imagine the program being made more sophisticated using the time of the day and the day of the week, if the air-conditioning is on or being attached to a smoke alarm.

We will look at different types of control statements, but first we need to introduce boolean types and comparison operators.

# Booleans

Before starting with control statements, we need to introduce booleans.
A Boolean is a type of variable that can take on one of two values - true or false.


```python
a = True
print(a)

a = False
print(a)
```

    True
    False


Booleans are used extensively in control statements.

# Comparison operators

We often want to check in a program how two variables are related to each other, for example if one is less than the other, or if two variables are equal. We do this with 'comparison operators', such as `<`, `<=`, `>`, `>=` and `==`. 

Below is an example checking if a number `a` is less than or greater than a number `b`:


```python
a = 10.0
b = 9.9
print(a < b)
print(a > b)
```

    False
    True


Equality is checked using '`==`', and '`!=`' is used to test if two variables are not equal. Below are some examples to read through.


```python
a = 14
b = -9
c = 14

# Check if a is equal to b 
print("Is a equal to b?")
print(a == b)

# Check if a is equal to c 
print("Is a equal to c?")
print(a == c)

# Check if a is not equal to c 
print("Is a not equal to c?")
print(a != c)

# Check if a is less than or equal to b 
print("Is a less than or equal to b?")
print(a <= b)

# Check if a is less than or equal to c 
print("Is a less than or equal to c?")
print(a <= c)

# Check if two colours are the same
colour0 = 'blue'
colour1 = 'green'
print("Is colour0 the same as colour1?")
print(colour0 == colour1)
```

    Is a equal to b?
    False
    Is a equal to c?
    True
    Is a not equal to c?
    False
    Is a less than or equal to b?
    False
    Is a less than or equal to c?
    True
    Is colour0 the same as colour1?
    False


# Boolean operators

In the above we have only used one comparison at a time. Boolean operators allow us to 'string' together multiple checks using the operators '`and`', '`or`' and '`not`'.
The operators '`and`' and '`or`' take a boolean on either side, and the code
```python
X and Y
```
will evaluate to `True` if `X` *and* `Y` are both true, and otherwise will evaluate to `False`. The code
```python
X or Y
```
will evaluate to `True` if `X` *or* `Y` is true, and otherwise will evaluate to `False`.
Here are some examples:


```python
# If 10 < 9 (false) and 15 < 20 (true) -> false
print(10 < 9 and 15 < 20)
```

    False



```python
# Check if 10 < 9 (false) or 15 < 20 (true) -> true
print(10 < 9 or 15 < 20)
```

    True


The meaning of the statement becomes clear if read it left-to-right.

Note that the comparison operators (`>=`, `<=`, `<` and `>`) are evaluated before the Boolean operators (`and`, `or`).

In Python, the '`not`' operator negates a statement, e.g.:


```python
# Is 12 *not* less than 7 -> true
a = 12
b = 7
print(not a < b)
```

    True


Only use '`not`' when it makes a program easy to read. For example, the following is not good practice.


```python
print(not 12 == 7)
```

    True


Better is


```python
print(12 != 7)
```

    True


Here is a double-negation, which is very cryptic (and poor programming):


```python
print(not not 12 == 7)
```

    False


## Multiple comparison operators

The examples so far use at most two comparison operators. In some cases we might want to perform more checks. We can control the order of evaluation using brackets. For example, if we want to check if a number is strictly between 100 and 200, or between 10 and 50:


```python
value = 150.5
print ((value > 100 and value < 200) or (value > 10 and value < 50)) 
```

    True


The two checks in the brackets are evaluated first (each evaluates to `True` or `False`), and then the '`or`' checks if one of the two is true.

# Control statements

Now that we've covered comparison, we are ready to look at control statements. These are a central part of computing. Here is a control statement in pseudo code:

    if A is true
        Perform task X (only)
    else if B is true
        Perform task Y (only)
    else   
        Perform task Z (only)

The above is an 'if' statement. Another type of control statement is

    do task X 10 times
    
We make this concrete below with some examples.

## `if` statements

Below is a simple example that demonstrates the Python syntax for an if-else control statement. 

Let's say we are writing a simple banking software. An account holder in our bank would like to withdraw money from their bank account. We need to check if their `balance` is greater than amount of money they would like to `withdraw`. If `balance > withdraw` then they can go ahead with the transaction and we reduce the balance by the withdrawn amount. We also need to display a message with their balance after withdrawing the money. On the other hand, if the `balance` is less than `withdraw`, we can't allow them to withdraw any money and this transaction will fail with the corresponding error message. Finally, we need to tell the account holder if the `balance` and `withdraw` are the same and it leads to zero cash left in the bank account. This is an important case and we need to let the user know that they have no money left in their account. Let's look at these cases:


```python
balance = 1000.0  # Initial balance

withdraw = 400.0

new_balance = balance - withdraw

if new_balance > 0.0:  
    balance = new_balance
    print('The new balance after withdrawal of ' \
          '{:.2f} is {:.2f}'.format(withdraw, balance))
elif new_balance == 0.0:  
    balance = new_balance
    print('You have no more money left after withdrawal of ' \
          '{:.2f} is {:.2f}'.format(withdraw, balance))
else: 
    print('Insufficient funds for transaction withdrawal of ' \
          '{:.2f} when balance is {:.2f}'.format(withdraw, balance))

# Print new x value
print("New balance is:", balance)
```

    The new balance after withdrawal of 400.00 is 600.00
    New balance is: 600.0


Try changing the value of `withdraw` and re-running the cell to see the different paths the code can follow.

We now dissect the control statement example. The control statement begins with an `if`, followed by the expression to check, followed by '`:`'
```python
if new_balance > 0.0:  
```
Below that is a block of code, indented by four spaces, that is executed if the check (`new_balance > 0.0`) is true:
````python
    balance = new_balance
    print('The new balance after withdrawal of ' \
          '{:.2f} is {:.2f}'.format(withdraw, balance))
````
and in which case the program will then move beyond the end of the control statement. If the check evaluates to false, then the `elif` (else if) check  
```python
elif new_balance == 0.0:  
    balance = new_balance
    print('You have no more money left after withdrawal of ' \
          '{:.2f} is {:.2f}'.format(withdraw, balance))
```      
is performed, and if true '`print('You have no money left after withdrawal')`' is executed and the control block is exited. The code following the `else` statement is executed
```python
else:
    print('Insufficient funds for transaction withdrawal of ' \
          '{:.2f} when balance is {:.2f}'.format(withdraw, balance)
```
if none of the preceding statements were true.

## Check if a number is divisible by 2 and 3, or print if is divisible by either 2 or 3



```python
## Check if a number is divisible by 2 and 3, or print if is divisible by either 2 or 3

n = 6

if (n % 2 == 0 and n % 3 == 0):
    print(n, "is divisible by 2 and 3")
elif(n % 2 == 0):
    print(n, "is divisible only by 2")
elif(n % 3 == 0):
    print(n, "is divisible only by 3")
else:
    print(n, "is not divisible by either 2 or 3")
```

    6 is divisible by 2 and 3


## The Taylor series expansion for cos(x) is

$$ \cos(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!}+ \dots = \sum_{n = 0}^{\infty}\frac{(-1)^n}{(2n)!}x^{2n}$$


```python
import math
import numpy as np

x = 0.5

cos = 1
n = 12
j = 1
for i in np.arange(2, n, 2):
    print(i)
    j *= -1
    cos += j * x**i/math.factorial(i)

print(cos)
print(math.cos(x))
assert(round(cos - math.cos(x), 5) == 0.0)
```

    2
    4
    6
    8
    10
    0.8775825618898637
    0.8775825618903728


## Check if a number is prime



```python
import numpy as np

n = 47

# Assume that n is prime
n_is_prime = True

# Check if n can be divided by m, where m ranges from 2 to n (excluding n)
for m in np.arange(2, n):
    if n % m == 0:  # This is true if the remainder for n/m is equal to zero
        # We've found that n is divisable by m, so it can't be a prime number. 
        # No need to check for more values of m, so set n_is_prime = False and
        # exit the 'm' loop.
        n_is_prime = False

#  If n is prime, print to screen        
if n_is_prime:
    print(n, "is prime")
else:
    print(n, "is not prime")
```

    47 is prime


## Iterate through a loop and find the optimal solution using an `if` condition


```python
# Iterate through angles using range and identify the optimum angle

# Import module
import math # trignometric functions
import sys # maximum int
import matplotlib.pyplot as plt
%matplotlib inline

# Assign variables
mu = 0.75   # friction coefficient
weight = 25 # Weight of the block in kN
theta = 45  # angle in degrees

# Create an empty list of forces
forces = []

# Create a list of angles from 0 to 90, arange(0, 91, 1)
angles = np.arange(91)

min_force = sys.maxsize
min_theta = 0

# Iterate through all angles
for theta in angles:
    # Compute pulling force: F = (mu * W) / (cos(theta) + mu * sin(theta))
    force = (mu * weight) / (math.cos(math.radians(theta)) +
                                       mu * math.sin(math.radians(theta)))

    forces.append(force)
    # If the minimum force is greater than the current force, replace with current force
    if min_force > force:
        min_force = force
        min_theta = theta

# Plot angles vs forces
plt.plot(angles, forces)
print(min_theta, min_force)
```

    37 15.000038671163749



    
![png](01_control_statements_solutions_files/01_control_statements_solutions_34_1.png)
    


## Loops: `break` and `continue`

### `break`


The `break` statement is used to terminate the loop prematurely when a certain condition is met. When break statement is encountered inside the body of the loop, the current iteration stops and program control immediately jumps to the statement following the loop. 



```python
# Check if we have a negative number in our list of numbers

numbers = [5, 11, 18, 4, 3, -8, 7, 0, -2, 1, 6, 3, -19, 21]

for number in numbers:
    print(number)
    if number < 0:
        print("The list has a negative number")
        break
```

    5
    11
    18
    4
    3
    -8
    The list has a negative number



Sometimes we want to break out of a `for`. Maybe in a `for` loop we can check if something is true, and then exit the loop prematurely. Below is a program for finding prime numbers that uses a `break` statement. Take some time to understand what it does. It might be helpful to add some print statements to understand the flow.


```python
import numpy as np
N = 50  # Check numbers up 50 for primes (excludes 50)

# Loop over all numbers from 2 to 50 (excluding 50)
for n in np.arange(2, N):

    # Assume that n is prime
    n_is_prime = True

    # Check if n can be divided by m, where m ranges from 2 to n (excluding n)
    for m in np.arange(2, n):
        if n % m == 0:  # This is true if the remainder for n/m is equal to zero
            # We've found that n is divisable by m, so it can't be a prime number. 
            # No need to check for more values of m, so set n_is_prime = False and
            # exit the 'm' loop.
            n_is_prime = False
            break

    #  If n is prime, print to screen        
    if n_is_prime:
        print(n)
```

    2
    3
    5
    7
    11
    13
    17
    19
    23
    29
    31
    37
    41
    43
    47


Try modifying the code for finding prime numbers such that it finds the first $N$ prime numbers (since you do not know how many numbers you need to check to find $N$ primes).

### `continue`

Sometimes we want to go prematurely to the next iteration in a loop, skipping the remaining code.
For this we use `continue`. Here is an example of the prime number calculator, except we now exit the loop if the number is divisible by 2 (even). If it is divisible by 2 it moves to the next value. If it is not divisible by 2 it advances the loop. 


```python
import numpy as np

N = 50  # Check numbers up 50 for primes (excludes 50)

# Loop over all numbers from 2 to 50 (excluding 50)
for n in np.arange(2, N):

    # Assume that n is prime
    n_is_prime = True
    
    # Check if n is divisible by 2
    if n % 2 == 0 and n !=2:
        continue

    # Check if n can be divided by m, where m ranges from 2 to n (excluding n)
    for m in np.arange(2, n):
        if n % m == 0:  # This is true if the remainder for n/m is equal to zero
            # We've found that n is divisable by m, so it can't be a prime number. 
            # No need to check for more values of m, so set n_is_prime = False and
            # exit the 'm' loop.
            n_is_prime = False
            break

    #  If n is prime, print to screen        
    if n_is_prime:
        print(n)
```

    2
    3
    5
    7
    11
    13
    17
    19
    23
    29
    31
    37
    41
    43
    47


It finds factors for numbers between 2 to 10. Now for the fun part. We can add an additional else block which catches the numbers which have no factors and are therefore prime numbers:


```python
import numpy as np
for n in np.arange(2, 10):
    for x in np.arange(2, n):
         if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else: #nobreak
        # loop fell through without finding a factor
        print(n, 'is a prime number')
```

    2 is a prime number
    3 is a prime number
    4 equals 2 * 2
    5 is a prime number
    6 equals 2 * 3
    7 is a prime number
    8 equals 2 * 4
    9 equals 3 * 3


### Continue

The continue statement is used to move ahead to the next iteration without executing the remaining statement in the body of the loop. 



```python
# Sum of all positive numbers in the list

numbers = [5, 11, 18, 4, 3, -8, 7, 0, -2, 1, 6, 3, -19, 21]

sum = 0
for number in numbers:
    if number < 0:
        continue
    sum += number
    print("Adding", number, "new sum", sum)

print(sum)
```

    Adding 5 new sum 5
    Adding 11 new sum 16
    Adding 18 new sum 34
    Adding 4 new sum 38
    Adding 3 new sum 41
    Adding 7 new sum 48
    Adding 0 new sum 48
    Adding 1 new sum 49
    Adding 6 new sum 55
    Adding 3 new sum 58
    Adding 21 new sum 79
    79


## Bisection approach for computing the angle for a given force



```python
# Import modules
import math
import numpy as np

# Initial guess
theta1 = 60
theta2 = 90

# Assign variables
mu = 0.75   # friction coefficient
weight = 25 # Weight of the block in kN
force = 17.5 # kN

theta = 0  # angle in degrees

# Set a tolerance
tolerance = 1e-5

# Iterate to a maximum of 1000 iterations
max_iterations = 100

iterations = 0

for i in np.arange(max_iterations):

    # Compute forces for theta1 and theta2
    delta1 = force - (mu * weight) / (math.cos(math.radians(theta1)) +
                                       mu * math.sin(math.radians(theta1)))
    
    delta2 = force - (mu * weight) / (math.cos(math.radians(theta2)) +
                                       mu * math.sin(math.radians(theta2)))
    
    # Compute the mid-value of theta
    theta = (theta1 + theta2)/2
    
    # Calculate the difference delta for the mid-theta   
    delta = force - (mu * weight) / (math.cos(math.radians(theta)) +
                                       mu * math.sin(math.radians(theta)))
    
    if((delta * delta1) > 0):
        theta1 = theta
    else:
        theta2 =  theta
    
    # Final values at the end of iterations
    iterations = i
    
    if (abs(delta) <= tolerance):
        break

else: #No break
    print("Solution did not converge!")
    
final_force = (mu * weight) / (math.cos(math.radians(theta)) +
                                       mu * math.sin(math.radians(theta)))

print('After', iterations, 'iterations the angle is {:.6f} deg, \
which gives a force of {:.10f} kN at a tolerance of {:.2E}'.format(theta, final_force, tolerance))
```

    After 14 iterations the angle is 67.872620 deg, which gives a force of 17.5000005229 kN at a tolerance of 1.00E-05


Note that the initial guess of 60 and 90 gave a result of 67.87 degrees, but an initial guess of 0 would give a value of theta equals 5.8672 degrees.

# Forces on a sliding block with multiple friction angles


```python
# Import matplotlib for plotting
# Import math for trignometric functions
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
%matplotlib inline
# Set plot size
plt.rcParams['figure.figsize'] = [7.5, 5]

# Weight of the block
weight = 25  # kN, A typical pyramid block is 2500 kg
# Friction on the bottom plane
frictions = [0.25, 0.5, 0.75]
colors = ['rs', 'b*', 'go']
lines = ['r--', 'b-', 'g-.']
index = 0
for friction in frictions:
    force_min = sys.maxsize

    # Create an empty list of angles and forces
    angles = []
    forces = []

    # for angles between 0 and 90*
    for theta in np.arange(0, 90, 1):

        # Compute pulling force: F = (mu * W) / (cos(theta) + mu * sin(theta))
        force = (friction * weight) / (math.cos(math.radians(theta)) +
                                       friction * math.sin(math.radians(theta)))

        # Add to list of angles and forces
        angles.append(theta)
        forces.append(force)
        if force < force_min:
            force_min = force  # Minimum force
            theta_min = theta  # Pulling angle that yields the minimum force

    # Plot force and angles
    plt.plot(angles, forces, lines[index], label='mu = ' + str(friction))
    plt.plot(theta_min, force_min, colors[index])
    # Increase index for color and lines
    index += 1

# Configure labels and legends
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("pulling angle (degrees)", fontsize=14)
plt.ylabel("required pulling force (kN)", fontsize=14)
plt.legend(fontsize=12)

# Display plot
plt.show()
```


    
![png](01_control_statements_solutions_files/01_control_statements_solutions_50_0.png)
    



```python

```
