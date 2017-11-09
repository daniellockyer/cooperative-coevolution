#![feature(alloc_system)] extern crate alloc_system;
#[cfg(test)] #[macro_use] extern crate assert_approx_eq;
extern crate rand;
extern crate half;

macro_rules! exp { ($a: expr) => ($a.exp()) }
macro_rules! sqrt { ($a: expr) => ($a.sqrt()) }
macro_rules! sin { ($a: expr) => ($a.sin()) }
macro_rules! cos { ($a: expr) => ($a.cos()) }
macro_rules! abs { ($a: expr) => ($a.abs()) }
macro_rules! sq { ($a: expr) => ($a * $a) }
macro_rules! sum { ($n: expr, $a: expr) => ((0 .. $n).map($a).fold(0.0, |a, b| a + b)) }
macro_rules! product { ($n: expr, $a: expr) => ((0 .. $n).map($a).fold(1.0, |a, b| a * b)) }

use rand::{thread_rng, Rng};
use half::f16;
use std::{env, f64};
use std::f64::consts::{PI, E};

const TARGET_LEN: usize = 16;

trait Function {
    fn calc(&self, x: Vec<f64>) -> f64;
}

struct Rastrigin;
struct Schwefel;
struct Griewangk;
struct Ackley;
struct Rosenbrock;

impl Function for Rastrigin {
    fn calc(&self, x: Vec<f64>) -> f64 {
        let n = x.len();
        let sum: f64 = sum!(n, |i| sq!(x[i]) - 3.0 * cos!(2.0 * PI * x[i]));

        3.0 * (n as f64) + sum
    }
}

impl Function for Schwefel {
    fn calc(&self, x: Vec<f64>) -> f64 {
        let n = x.len();
        let sum: f64 = sum!(n, |i| x[i] * sin!(sqrt!(abs!(x[i]))));

        418.982887 * (n as f64) - sum
    }
}

impl Function for Griewangk {
    fn calc(&self, x: Vec<f64>) -> f64 {
        let n = x.len();
        let sum: f64 = sum!(n, |i| sq!(x[i]) / 4000.0);
        let product: f64 = product!(n, |i| cos!(x[i] / sqrt!(((i + 1) as f64))));

        1.0 + sum - product
    }
}

impl Function for Ackley {
    fn calc(&self, x: Vec<f64>) -> f64 {
        let n = x.len();
        let sum1: f64 = sum!(n, |i| sq!(x[i]));
        let sum2: f64 = sum!(n, |i| cos!(2.0 * PI * x[i]));

        20.0 + E
            - (20.0 * exp!(-0.2 * sqrt!(1.0 / (n as f64) * sum1)))
            - exp!(1.0 / (n as f64) * sum2)
    }
}

impl Function for Rosenbrock {
    fn calc(&self, x: Vec<f64>) -> f64 {
        100.0 * sq!(sq!(x[0]) - x[1]) + sq!(1.0 - x[0])
    }
}

#[derive(Debug)]
struct EvoPheno {
    val: u16,
    fitness: f64,
}

impl EvoPheno {
    fn new(t: u16) -> EvoPheno {
        EvoPheno {
            val: t,
            fitness: f64::MAX
        }
    }

    fn crossover(&self, other: &EvoPheno) -> EvoPheno {
        let mut new = self.val;
        let (low, high) = {
            let a = thread_rng().gen_range(0, TARGET_LEN);
            let b = thread_rng().gen_range(0, TARGET_LEN);

            if a < b {
                (a, b)
            } else {
                (b, a)
            }
        };

        for i in 0..TARGET_LEN {
            if i > low && i < high {
                let j_value = (other.val & (1 << i)) >> i;
                new = (new & (!(1 << i))) | (j_value << i);
            }
        }

        EvoPheno::new(new)
    }

    fn mutate(&self) -> EvoPheno {
        let mut new = self.val;

        for i in 0..TARGET_LEN {
            if rand::random::<f64>() < (1f64 / TARGET_LEN as f64) {
                new ^= 1 << i;
            }
        }

        EvoPheno::new(new)
    }
}

fn make_vec(size: u32, value: u16) -> Vec<f64> {
    (0..size).map(|_| f64::from(f16::from_bits(value))).collect()
}

fn tournament(population: &Vec<EvoPheno>) -> (usize, usize) {
    let a_index = thread_rng().gen_range(0, population.len()) as usize;
    let b_index = thread_rng().gen_range(0, population.len()) as usize;

    if population[a_index].fitness < population[b_index].fitness {
        (a_index, b_index)
    } else {
        (b_index, a_index)
    }
}

fn print_pop(iterations: u32, population: &Vec<EvoPheno>, show_pop: bool) {
    if show_pop {
        print!("{}[2J", 27 as char);
        for p in population {
            println!("{:?} = {}", make_vec(1, p.val), p.fitness);
        }
    }

    let mut lowest_fitness = f64::MAX;
    let mut best_value = f64::MAX;
    for p in population {
        if p.fitness < lowest_fitness {
            lowest_fitness = p.fitness;
            best_value = f64::from(f16::from_bits(p.val));
        }
    }
    println!("{},{},{}", iterations, best_value, lowest_fitness);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: coopco <crossover> <function>");
        std::process::exit(1);
    }

    let population_size = 100;
    let crossover: bool = args[1].parse().expect("First arg is crossover");

    let (function, dimensions): (Box<Function+'static>, u32) = match args[2].as_str() {
        "Ra" => (Box::new(Rastrigin), 20),
        "Sc" => (Box::new(Schwefel), 10),
        "Gr" => (Box::new(Griewangk), 10),
        "Ac" => (Box::new(Ackley), 10),
        "Ro" => (Box::new(Rosenbrock), 2),
        _ => {
            println!("Defaulting to Rastrigin");
            (Box::new(Rastrigin), 20)
        }
    };

    let mut population: Vec<EvoPheno> = (0..population_size).map(|_| {
        let mut p = EvoPheno::new(thread_rng().gen::<u16>());
        p.fitness = function.calc(make_vec(dimensions, p.val)).abs();
        p
    }).collect();

    let mut iterations = 0;
    println!("iteration,value,fitness");

    while iterations < 1000 {
        let (parent1_index, _) = tournament(&population);

        let mut child = if crossover {
            let (parent2_index, _) = tournament(&population);
            population[parent1_index].crossover(&population[parent2_index]).mutate()
        } else {
            population[parent1_index].mutate()
        };

        iterations += 1;
        child.fitness = function.calc(make_vec(dimensions, child.val)).abs();

        let (_, new_index) = tournament(&population);
        std::mem::replace(&mut population[new_index], child);

        print_pop(iterations, &population, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_approx_eq!(0.0, Rastrigin.calc([0.0; 20].to_vec()));
        assert_approx_eq!(0.0, Schwefel.calc([420.968746; 10].to_vec()), 1e-5f64);
        assert_approx_eq!(0.0, Griewangk.calc([0.0; 10].to_vec()));
        assert_approx_eq!(0.0, Ackley.calc([0.0; 30].to_vec()));
        assert_approx_eq!(0.0, Rosenbrock.calc([1.0; 2].to_vec()));
    }
}
