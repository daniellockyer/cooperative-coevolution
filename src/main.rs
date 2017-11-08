#![feature(alloc_system)] extern crate alloc_system;
#[macro_use] extern crate assert_approx_eq;
extern crate rand;
extern crate half;

mod func;

use rand::{thread_rng, Rng};
use half::f16;
use std::{env, f64};

const TARGET_LEN: usize = 16;

#[derive(Clone, Debug)]
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

        for i in 0..TARGET_LEN {
            if rand::random::<f64>() < 0.6 {
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

    if args.len() < 4 {
        eprintln!("Usage: coopco <population_size> <crossover> <function> <print>");
        std::process::exit(1);
    }

    let population_size: u32 = args[1].parse().expect("First arg is population_size");
    let crossover: bool = args[2].parse().expect("Second arg is crossover");
    let print: bool = args[3].parse().expect("Third arg is printing");

    let (function, dimensions): (Box<func::Function+'static>, u32) = match args[4].as_str() {
        "Ra" => (Box::new(func::Rastrigin), 20),
        "Sc" => (Box::new(func::Schwefel), 10),
        "Gr" => (Box::new(func::Griewangk), 10),
        "Ac" => (Box::new(func::Ackley), 10),
        "Ro" => (Box::new(func::Rosenbrock), 2),
        _ => {
            println!("Defaulting to Rastrigin");
            (Box::new(func::Rastrigin), 20)
        }
    };

    let mut population: Vec<EvoPheno> =
        (0..population_size).map(|_| EvoPheno::new(thread_rng().gen::<u16>())).collect();

    for p in &mut population {
        p.fitness = function.calc(make_vec(dimensions, p.val)).abs();
    }

    let mut iterations = 0;
    if print {
        println!("iteration,value,fitness");
    }

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

        if print {
            print_pop(iterations, &population, false);
        }
    }
}
