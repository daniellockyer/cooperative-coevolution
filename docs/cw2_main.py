import numpy
from copy import deepcopy

def rastrigin(xi):
    """
    USE K = 3
    """
    n = 20
    return 3 * n + (xi * xi - 3 * numpy.cos(2 * numpy.pi * xi)).sum(axis=1)

def schwefel(xi):
    """
    USE K = 9
    """
    n = 10
    return 418.9829 * n - (xi * numpy.sin(numpy.sqrt(numpy.abs(xi)))).sum(axis=1)

def griewangk(xi):
    """
    USE K = 10
    """
    n = numpy.zeros_like(xi)
    for i in range(len(xi)):
        n[i] = xi[i] / numpy.sqrt(i+1)
    return 1 + (xi * xi / 4000).sum(axis=1) - (numpy.cos(n)).prod(axis=1)

def ackley(xi):
    """
    USE K = 5
    """
    n = 1 / 30
    return 20 + numpy.e - 20 * numpy.exp(-0.2 * numpy.sqrt((xi * xi).sum(axis=1) * n)) - \
           numpy.exp((numpy.cos(2 * numpy.pi * xi)).sum(axis=1) * n)

def binary_create():
    x = numpy.random.rand(16)
    xc = x.copy()
    xc[x > 0.5] = 1
    xc[x < 0.5] = 0
    return xc

def val_transt(xi, k):
    """translate values"""
    p2 = numpy.power(2, numpy.fliplr([numpy.arange(k-15, k)])[0].astype(numpy.float64)) # gets the powers of 2
    return (-1)**xi[:,0] * numpy.dot(xi[:, 1:], p2)

def val_transt1(xi, k):
    """translate values"""
    p2 = numpy.power(2, numpy.fliplr([numpy.arange(k-15, k)])[0].astype(numpy.float64)) # gets the powers of 2
    return (-1)**xi[:,:,0] * numpy.dot(xi[:,:, 1:], p2)

def fit_prop_give_index(fitness):
    """
    fitness - array containing the fitness values of each stuff
    """
    fitness = numpy.abs(fitness)
    total_fitness = (fitness).sum()
    random_fitness = numpy.random.uniform(0, total_fitness)
    return numpy.argmax(fitness.cumsum() >= random_fitness)

def plot_GA(f, n, val, k):
    X = ga_cross(f, n, val, k)

def mutation(x1, x2, k, val):
    """
    crossover & bitflip
    """
    co = 0.6
    N, n = x1.shape
    mp = 1/ n

    A = numpy.random.randint(1, n-2)
    B = numpy.random.randint(A+1, n)
    child = x1
    t = numpy.random.random(N) < co
    child[t,A:B] = x2[t, A:B]

    child[numpy.random.random(child.shape) < mp] -= 1
    child = numpy.abs(child)
    done = 0
    while not done:
        c_o = numpy.abs(val_transt(child, k)) > val
        if (c_o.sum() == 0):
            done = 1
        else:
            child2 = child[c_o]
            child2[numpy.random.random(child2.shape) < mp] -= 1
            child[c_o] = numpy.abs(child2)
    return child

def ga_cross(f, n, val, k):
    x_old = ga_init_vals(n, val, k)
    fitness = numpy.zeros(100)
    fitness = f(val_transt1(x_old, k))
    max_fit = fitness[fitness.argmin()]
    min_fit = abs(fitness[fitness.argmax()])

    iterations = 10**5
    results = np.zeros(iterations) #max fitness
    x_new = numpy.zeros_like(x_old)

    #first try with just 2 children
    for i in range(iterations):
        fitness = numpy.abs(f(val_transt1(x_old, k)))
        elite = fitness.argmin() # index of the elite
        tre = deepcopy(x_old[elite])
        for j in range(50):
            #remake the pop from old pop
            A = fit_prop_give_index(min_fit - fitness) #so the closer you are to 0, the more chances there are
            B = fit_prop_give_index(min_fit - fitness)

            x_new[2*j] = mutation(x_old[A], x_old[B], k, val) #child1
            x_new[2*j+1] = mutation(x_old[A], x_old[B], k, val) #child2

        results[i] = fitness[elite]

        fitness_new = numpy.abs(f(val_transt1(x_new, k)))
        not_so_elite = fitness_new.argmax()
        x_new[not_so_elite] = tre
        x_old = x_new.copy()
        if i % 100 == 0:
            print(i, fitness[fitness.argmin()], fitness_new[fitness_new.argmin()])
            with open("{}.txt".format(f.__name__), 'a+') as dat:
                dat.write(numpy.str(results[i]) + "\n") # just in case

            del(dat)


    return results, x_old, fitness

def ccga(f, n, val, k):
    x = ga_init_vals(n, val, k)
    init_fit = numpy.zeros(100)
    for i in range(100):
        init_fit[i] = f(val_transt(x[i], k))
        print(init_fit[i])
    max_fit = init_fit[init_fit.argmin()]
    results = np.zeros(100000)

def ga_init_vals(n, val, k):
    """
    n - nr of vars,
    val - max value of var
    k point at which "the binary goes to decimal"""
    xi = numpy.zeros([100, n, 16])
    for j in range(100):
        for i in range(n):
            done = 0
            while not done:
                xi[j, i] = binary_create()
                if ((abs(val_transt(xi[j], k)) - val) <= 0).all():
                    done = 1
    return xi

def ccga_init_vals(n, val, k):
    """
    n - nr of vars,
    val - max value of var
    k point at which "the binary goes to decimal"""
    xi = numpy.zeros([100, n, n, 16])
    for j in range(100):
        for i in range(n):
            for ii in range(n):
                done = 0
                while not done:
                    xi[j, i, ii] = binary_create()
                    if ((abs(val_transt(xi[j, i], k)) - val) <= 0 ).all():
                        done = 1
    return xi