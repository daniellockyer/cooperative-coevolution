t = Number of generations
Cooperative or selfish:
    * Gi - growth rate
    * Ci - resource comsumption rate
Initial size of the group (small or large)

Coop + small
Coop + large
Selfish + small
Selfish + large

n_i = number of copies of genotype i in a single group

A clone requires a share of the group's resource influx, R.

the amount of resource that a group receives per time-step depends on its size, with a larger per capita amount of resource allocated to larger groups.

A group that is twice as larfe received 5% extra per capita.

magnitude of the hare of the total, r_i:

    r_i = (n_i * Gi * Ci) / Sum(n_j * Gj * Cj)) * R

genotype with the highest growth and consumption rates will receive the largest capita per share. a selfish genotype will receive more per capita resource than coop so coop geno will go extinct.

given the share received by a genotype, the number of individuals with that genotype:

    n_i(t + 1) = n_i(t) + (r_i / Ci) - (K * n_i(t))

number of clones depends on growth (where selfish wins) and consumption rates (where coop wins long-term). the final term represents a constant mortality rate

individuals carry a gene that determines the initial size of the group they join.

Overall algorithm:

1. Initialise a pool with N individuals
2. Group formation (aggregation) - assign individuals in the migrant pool to groups.
3. Reproduction - perform reproduction within groups for t time-steps as above.
4. Migrant pool formation (dispersal): Return the progeny of each group to the migrant pool.
5. Maintaining the global carrying capacity - rescale the migrant pool back to size N, retaining the proportion of individuals with each genotype.
6. Iteration - repeat from step 2 onwards for a number of generations, T.


Parameter settings:

Growth rate (coop) Gc = 0.018
Growth rate (self) Gs = 0.02
Consumption rate (coop) Cc = 0.1
Consumption rate (self) Cs = 0.2
Population size, N = 4000
Number of generations, T = 1000

----
from the web page:

Death rate = K = 0.1
R for small group = 4
R for large groups of 40 = 50

