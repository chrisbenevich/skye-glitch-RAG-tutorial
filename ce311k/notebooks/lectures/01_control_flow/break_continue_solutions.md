# Break continue (solution)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/01_control_flow/break_continue_solutions.ipynb)

 `continue`:
 A program to print sum of odd numbers between 0 to 100


```
import numpy as np

sum = 0

for i in np.arange(0, 100, 1):
  if i % 2 == 0:
    continue

  sum = sum + i

print(sum)

```

    2500


`break` statement to see if a list of numbers has a perfect square


```
import math

numbers = [1, 5, 0, 25, 23, 12, 7, 51]

for number in numbers:
  sqrt_num = math.sqrt(number)
  if (sqrt_num % 1 == 0) and number !=1 and number != 0:
    print("The list has a perfect square number:", number)
    break
else: #nobreak
  print("The list has no perfect squares")
```

    The list has a perfect square number: 25


Prime numbers


```
import numpy as np

N = 50  # Check numbers up 50 for primes (excludes 50)
primes = []
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

    #  If n is prime, add to list of primes
    if n_is_prime:
        primes.append(n)

# Primes
print(primes)
```

    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]



```

```
