#### About

This is some example code for doing MIRT (multidimensional item response theory) simulations in Python using PyMC.

#### Requirements
[PyLab](http://www.scipy.org/PyLab)

[PyMC](https://github.com/pymc-devs/pymc)

#### Usage

Run simulation.py, toggling the ```generating``` variable to switch between generating mode (where we assume we know the underlying learner proficiencies, and generate a set of responses) and non-generating mode (when all we know is the responses, and we want to infer the underlying proficiencies).
