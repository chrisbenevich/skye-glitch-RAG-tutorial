# Lecture 01b: Example of control statement

**Exercise:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/01_control_flow/01_friction_sliding_block.ipynb)
**Solution:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kks32-courses/ce311k/blob/main/notebooks/lectures/01_control_flow/01_friction_sliding_block_solutions.ipynb)


<iframe width="560" height="315" src="https://www.youtube.com/embed/pOOkpfQGkaw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/K3PPABjazHI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Forces on a sliding block

![forces on a sliding block](https://raw.githubusercontent.com/kks32-courses/ce311k/master/notebooks/lectures/01_control_flow/block-forces.png)

Given a weight $W = 25 kN$, friction coefficient $\mu = 0.75$, and angle of inclination $\theta = 45^o$, calculate the force required to pull the block.

$$
F = \frac{\mu W}{\cos \theta + \mu \sin \theta} = \frac{\mu mg}{\cos \theta + \mu \sin \theta}
$$


Python has a lot of useful standard library functions which are defined in modules. A Python code can gain access to the code in another module by the process of importing it. The import statement is the most common way of invoking the import machinery, but it is not the only way. We need to import a module called `math`, which has predefined trignometric functions using `import math`, this makes available all the functions in `math` module. To use the `cos` function in `math` module, we do `math.cos(angle)`


```python
# Import math module for trignometric functions
import math

# Assign variables
mu = 0.75   # friction coefficient
weight = 25 # Weight of the block in kN
theta = 35  # angle in degrees

# Compute pulling force: F = (mu * W) / (cos(theta) + mu * sin(theta))
force = (mu * weight) / (math.cos(math.radians(theta)) +
                                       mu * math.sin(math.radians(theta)))

print("Force is {:.2f} kN".format(force))
```

    Force is 15.01 kN


## Lists

A list is a sequence of data. An 'array' in most other languages is a similar concept, but Python lists are more general than most arrays as they can hold a mixture of types. A list is constructed using square brackets. To create a list of Pixar movies `['Toy Story', 'Monsters Inc', 'Finding Nemo', 'The Incredibles', 'Cars']`:


```python
pixar = ["Toy Story", "Monsters Inc", "Finding Nemo", "The Incredibles", "Cars"]

print(pixar)
print(len(pixar))
```

    ['Toy Story', 'Monsters Inc', 'Finding Nemo', 'The Incredibles', 'Cars']
    5


An empty list is created by `[]`


```python
empty_list = []
```

To add an item to list use `list.append(item)`. Let's add a new Pixar movie: `coco`.


```python
pixar.append("Coco")
print("Number of Pixar movies: {}".format(len(pixar)))
```

    Number of Pixar movies: 6


### Iterating through a list


Looping over each item in a list (or more generally a sequence) is called *iterating*. We iterate over the movies in the pixar list using the syntax:


```python
for movie in pixar:
    print(movie)
```

    Toy Story
    Monsters Inc
    Finding Nemo
    The Incredibles
    Cars
    Coco


### Create a list of possible angles 

`angles` should store a list of ```[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]```


```python
# A list of possible angles
angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
print(angles)
```

    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


### Iterate through all the angles and calculate the force for each of those angles

print `(angle, force)` for each angle.


```python
# Iterate through angles

# Import math module for trignometric functions
import math

# Assign variables
mu = 0.75   # friction coefficient
weight = 25 # Weight of the block in kN
theta = 45  # angle in degrees

# Create a list of angles
angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

for theta in angles:
    # Compute pulling force: F = (mu * W) / (cos(theta) + mu * sin(theta))
    force = (mu * weight) / (math.cos(math.radians(theta)) +
                                       mu * math.sin(math.radians(theta)))

    print(theta, force)
```

    0 18.75
    10 16.81548164247548
    20 15.674535080088216
    30 15.108473962598113
    40 15.02241163085036
    50 15.402675952323367
    60 16.31116940054498
    70 17.911908792668115
    80 20.553486370758943
    90 24.999999999999996


### Plot the forces and angles

We will be using a module called `matplotlib` using `import matplotlib.pyplot as plt` to plot a list of angles and forces. The additional `as` part in the `import module` functions helps us keeping our code concise. Instead of typing `matplotlib.pyplot.plot()` we can just type `plt.plot()`. You may need to include `%matplotlib inline` to enable inline plotting in Jupyter notebook, include inline after you import `matplotlib`.

Let's create an empty list of forces using `[]`. At each iteration `append` the force to the `forces` list.


```python
# Iterate through angles and plot

# Import math module for trignometric functions
import math

# Import plotting functions from matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# Assign variables
mu = 0.75   # friction coefficient
weight = 25 # Weight of the block in kN
theta = 45  # angle in degrees

# Create a list of angles
angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# Create an empty list of forces
forces = []

for theta in angles:
    # Compute pulling force: F = (mu * W) / (cos(theta) + mu * sin(theta))
    force = (mu * weight) / (math.cos(math.radians(theta)) +
                                       mu * math.sin(math.radians(theta)))

    forces.append(force)


# Plot angles vs forces
plt.xlabel("Angles (degrees)")
plt.ylabel("Force (kN)")
plt.plot(angles, forces)

```




    [<matplotlib.lines.Line2D at 0x7f363d2c1bd0>]




    
![png](01_friction_sliding_block_solutions_files/01_friction_sliding_block_solutions_18_1.png)
    


### Update the angles list to increment every 1 degree from 0 to 90

Create an updated list of angles so that `angles = [0, 1, 2, ..., 90]`. A simple function to do that is `np.arange(stop)`. This function is available in a library called `numpy` and we will import it as `import numpy as np`.

`stop`: Number of integers (whole numbers) to generate, starting from zero up to but not including `stop`.

e.g., `np.arange(3)` yields a sequence of `[0, 1, 2]`.

`np.arange([start], stop[, step])`

`start`: Starting number of the sequence.
`stop` : Generate numbers up to, but not including this number.
`step` : Difference between each number in the sequence.

e.g., `np.arange(2, 10, 2)` yields a list of `[2, 4, 6, 8]`.


```python
# Iterate through angles using range and plot

# Import math module for trignometric functions
import math
import numpy as np
# Import plotting functions from matplotlib
import matplotlib.pyplot as plt

# Assign variables
mu = 0.75   # friction coefficient
weight = 25 # Weight of the block in kN
theta = 45  # angle in degrees

# Create an empty list of forces
forces = []

# Create a list of angles from 0 to 90, range(0, 91, 1)
angles = list(np.arange(0,91,13.50))

# Iterate through all angles
for theta in angles:
    # Compute pulling force: F = (mu * W) / (cos(theta) + mu * sin(theta))
    force = (mu * weight) / (math.cos(math.radians(theta)) +
                                       mu * math.sin(math.radians(theta)))

    forces.append(force)


# Plot angles vs forces
plt.plot(angles, forces)
```




    [<matplotlib.lines.Line2D at 0x7fed54abfaf0>]




    
![png](01_friction_sliding_block_solutions_files/01_friction_sliding_block_solutions_20_1.png)
    



```python

```
