better method for function optimisation by modelling the coevolution of cooperation species

we can think of the solution to a function optimization problem, with N parameters, as N subgroups

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
