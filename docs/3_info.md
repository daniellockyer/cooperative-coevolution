better method for function optimisation by modelling the coevolution of cooperation species

we can think of the solution to a function optimization problem, with N parameters, as N subgroups

CCGA = cooperative coevolutionary genetic algorithms

Traditional GA:

```
gen = 0
Pop(gen) = randomly initialized population
evaluate fitness of each individual in Pop(gen)

while termination condition = false do begin
    gen = gen + 1
    select Pop(gen) from Pop(gen − 1) based on fitness
    apply genetic operators to Pop(gen)
    evaluate fitness of each individual in Pop(gen)
    end
```

CCGA-1:

```
gen = 0
for each species s do begin
    Pops(gen) = randomly initialized population
    evaluate fitness of each individual in Pops(gen)
    end

while termination condition = false do begin
    gen = gen + 1
    for each species s do begin
        select Pops(gen) from Pops(gen − 1) based on fitness
        apply genetic operators to Pops(gen)
        evaluate fitness of each individual in Pops(gen)
        end
    end
```

Parameters:

representation: binary (16 bits per variable)
selection: fitness proportionate
fitness scaling: scaling window technique (width of 5)
elitist strategy: single copy of best individual preserved
genetic operators two-point crossover and bit-flip mutation
mutation probability: 1/chromlength
crossover probability: 0.6
population size: 100

Results in paper are average over 50 runs.

#### Rastrigin function

    f(x) = 3n + Sum(x^2 - 3cos(2 * PI * x))

where n = 20 and -5.12 <= x <= 5.12

#### Schwefel function

    f(x) = 418.9829n + Sum(x * sin(sqrt(mod(x))))

where n = 10 and -500 <= x <= 500

This function has a second-best minimum far away from the global minimum to trap algorithms.

#### Griewangk function

    f(x) = 1 + Sum(x^2 / 4000) - Product(cos(x/sqrt(i)))

where n = 10 and -600 <= x <= 600

#### Ackley function

    f(x) = 20 + e - 20exp(-0.2 * sqrt(1/n * Sum(x^2))) - exp(1/n * Sum(cos(2 * PI * x)))

where n = 30 and -30 <= x <= 30

---

Max termination iterations at 100,000. The lower the fitness, the better

https://www.mathworks.com/matlabcentral/fileexchange/46836-optimization-griewangk-function-by-particle-swarm-optimization?
