import numpy

def rastrigin(x):
    n = 20
    return 3 * n + (x * x - 3 * numpy.cos(2 * numpy.pi * x))

def schwefel(x):
    n = 10
    # Huh, the paper says + but online says -.
    return 418.9829 * n + (x * numpy.sin(numpy.sqrt(numpy.abs(x))))

print(rastrigin(0))
print(schwefel(0))
print(schwefel(1))
print(schwefel(2))
