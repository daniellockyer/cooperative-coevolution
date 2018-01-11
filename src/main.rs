#![feature(alloc_system)] extern crate alloc_system;
#[cfg(test)] #[macro_use] extern crate assert_approx_eq;
extern crate bit_vec;
extern crate rand;

use bit_vec::BitVec;
use rand::{Rng, thread_rng};
use std::{env, u16, f64, usize};
use std::f64::consts::{PI, E};
use std::fs::File;
use std::io::Write;

macro_rules! exp { ($a: expr) => ($a.exp()) }
macro_rules! sqrt { ($a: expr) => ($a.sqrt()) }
macro_rules! sin { ($a: expr) => ($a.sin()) }
macro_rules! cos { ($a: expr) => ($a.cos()) }
macro_rules! abs { ($a: expr) => ($a.abs()) }
macro_rules! sq { ($a: expr) => ($a * $a) }
macro_rules! sum { ($n: expr, $a: expr) => ((0 .. $n).map($a).fold(0.0, |a, b| a + b)) }
macro_rules! product { ($n: expr, $a: expr) => ((0 .. $n).map($a).fold(1.0, |a, b| a * b)) }
macro_rules! mapc { ($n: expr, $body: expr) => ((0 .. $n).map($body).collect()) }

const TARGET_LEN: usize = 16;
const POPULATION_SIZE: usize = 100;
const MAX_ITERATIONS: u32 = 1000;
const MAX_FUNCTION_EVALUATIONS: u32 = 100_000;

type SmBitVec = BitVec<u16>;

#[derive(Debug, Clone)]
struct EvoPheno {
    val: SmBitVec,
    fitness: f64,
}

impl EvoPheno {
    fn new(bv: SmBitVec) -> EvoPheno {
        EvoPheno {
            val: bv,
            fitness: f64::MAX
        }
    }

    fn new_from_vec(t: &Vec<u16>) -> EvoPheno {
        let mut bv: SmBitVec = BitVec::from_elem(TARGET_LEN * t.len(), false);

        for i in 0..bv.len() {
            let j_value = (t[i / TARGET_LEN] & (1 << i)) >> i;
            bv.set(i, j_value == 1);
        }

        EvoPheno::new(bv)
    }

    fn mutate(&mut self) {
        let mut rand = thread_rng();
        let len = self.val.len();
        let prob = 1.0 / len as f64;

        for i in 0..len {
            if rand.next_f64() <= prob {
                let cur_val = self.val[i];
                self.val.set(i, !cur_val);
            }
        }
   }

    fn clever_mutate(&mut self, iteration: u32) {
        let mut rand = thread_rng();
        let len = self.val.len();

        for i in 0..len {
            let prob = 0.00875 * (cos!(PI * (i / len) as f64 - PI * 
                (iteration / MAX_FUNCTION_EVALUATIONS) as f64)) + 0.07125;

            if rand.next_f64() <= prob {
                let cur_val = self.val[i];
                self.val.set(i, !cur_val);
            }
        }
    }
}

fn select_new_population(population: &[EvoPheno]) -> Vec<EvoPheno> {
    let mut limits = Vec::new();
    let mut new_population = Vec::new();
    let mut limit_sum = 0.0;

    let total_fitness = sum!(POPULATION_SIZE, |i| population[i].fitness);

    for i in 0..POPULATION_SIZE {
        limit_sum += 1.0 / (population[i].fitness / total_fitness);
        limits.push(limit_sum);
    }

    for _ in 0..POPULATION_SIZE {
        let mut j = 0;
        let random_number = thread_rng().gen_range(0.0, limit_sum);

        while random_number > limits[j] {
            j += 1;
        }

        new_population.push(population[j].clone());
    }

    new_population
}

fn recombine(population: &[EvoPheno]) -> Vec<EvoPheno> {
    let mut new_population = Vec::new();
    let mut remaining_population = POPULATION_SIZE;

    for i in 0..POPULATION_SIZE/2 {
        let index = i * 2;

        let random_number = thread_rng().gen_range(0, remaining_population);
        new_population.push(population[random_number].clone());
        remaining_population -= 1;

        let random_number = thread_rng().gen_range(0, remaining_population);
        new_population.push(population[random_number].clone());
        remaining_population -= 1;

        if thread_rng().next_f64() <= 0.6 {
            let len = TARGET_LEN;
            let (low, high) = {
                let a = thread_rng().gen_range(0, len);
                let b = thread_rng().gen_range(0, len);

                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            };

            for i in low..high {
                let temp_a = new_population[index].val[i];
                let temp_b = new_population[index + 1].val[i];

                new_population[index].val.set(i, temp_b);
                new_population[index + 1].val.set(i, temp_a);
            }
        }
    }

    new_population
}

fn real_val(bounds: f64, value: u16) -> f64 {
    -bounds + (f64::from(value) / f64::from(u16::MAX) * bounds * 2.0)
}

fn make_vec(bv: &SmBitVec, bounds: f64) -> Vec<f64> {
    let mut a = Vec::new();
    for i in bv.blocks() {
        a.push(real_val(bounds, i));
    }
    a
}

fn convert(r: &Vec<&SmBitVec>, bounds: f64) -> Vec<f64> {
    let mut a = Vec::new();
    for j in r {
        a.extend(make_vec(j, bounds));
    }
    a
}

fn run_ga(function: &Function) {
    let dimensions = function.dimensions();
    let bounds = function.bounds();

    let mut ga_file = File::create("ga.csv").unwrap();
    let mut ccga1_file = File::create("ccga1.csv").unwrap();
    let mut ccga4_file = File::create("ccga1v.csv").unwrap();

    for r in 0..10 {
        println!("GA-{}", r);
        let mut best_found_cost = f64::MAX;
        let mut fitness_data = Vec::new();
        let mut function_evaluations = 0;

        let mut population: Vec<EvoPheno> = mapc!(POPULATION_SIZE, |_| {
            let mut p = EvoPheno::new_from_vec(&mapc!(dimensions, |_|
                thread_rng().gen::<u16>()));
            p.fitness = function.calc(&make_vec(&p.val, bounds));
            function_evaluations += 1;
            p
        });

        while function_evaluations < MAX_FUNCTION_EVALUATIONS {
            population = select_new_population(&population);
            population = recombine(&population);

            for i in &mut population {
                i.mutate();
                i.fitness = function.calc(&make_vec(&i.val, bounds));
                function_evaluations += 1;

                if i.fitness < best_found_cost {
                    best_found_cost = i.fitness;
                }
            }

            fitness_data.push(best_found_cost);
        };
        let output: Vec<String> = fitness_data.iter().map(|n| n.to_string()).collect();
        writeln!(ga_file, "{}", output.join(",")).unwrap();
    }

    for r in 0..15 {
        println!("CCGA1-{}", r);
        let mut best_species = Vec::new();
        let mut fitness_data = Vec::new();
        let mut function_evaluations = 0;

        let mut population: Vec<Vec<EvoPheno>> =
            (0..dimensions).map(|_| mapc!(POPULATION_SIZE, |_|
                EvoPheno::new_from_vec(&mapc!(1, |_|
                    thread_rng().gen::<u16>())))).collect();
        let pop2 = population.clone();

        for d in 0..dimensions as usize {
            for individual in &mut population[d] {
                individual.fitness = function.calc(&convert(&mapc!(dimensions as usize,
                |dp| {
                    if dp == d {
                        &individual.val
                    } else {
                        &pop2[dp][thread_rng().gen_range(0, POPULATION_SIZE)].val
                    }
                }), bounds));
                function_evaluations += 1;
            }
        }

        for d in 0..dimensions as usize {
            let rand = thread_rng().gen_range(0, POPULATION_SIZE);
            best_species.push(population[d][rand].val.clone());
        }

        let mut best_found_cost =
            function.calc(&convert(&mapc!(dimensions as usize,
                |i| &best_species[i]), bounds));
        function_evaluations += 1;

        while function_evaluations < MAX_FUNCTION_EVALUATIONS {
            for d in 0..dimensions as usize {
                population[d] = select_new_population(&population[d]);
                population[d] = recombine(&population[d]);

                for i in &mut population[d] {
                    i.mutate();
                }

                let mut best_individual = 0;
                let mut best_cost = f64::MAX;

                for (i, individual) in population[d].iter_mut().enumerate() {
                    individual.fitness = function.calc(&convert(&mapc!(dimensions as usize,
                    |dp| {
                        if dp == d {
                            &individual.val
                        } else {
                            &best_species[dp]
                        }
                    }), bounds));
                    function_evaluations += 1;

                    if individual.fitness < best_cost {
                        best_cost = individual.fitness;
                        best_individual = i;
                    }
                }

                best_species[d] = population[d][best_individual].val.clone();

                let fitness = function.calc(&convert(&mapc!(dimensions as usize,
                    |i| &best_species[i]), bounds));
                function_evaluations += 1;
                if fitness < best_found_cost {
                    best_found_cost = fitness;
                }
            }
            fitness_data.push(best_found_cost);
        }
        let output: Vec<String> = fitness_data.iter().map(|n| n.to_string()).collect();
        writeln!(ccga1_file, "{}", output.join(",")).unwrap();
    }

    for r in 0..15 {
        println!("CCGA1-Variable-{}", r);
        let mut best_species = Vec::new();
        let mut fitness_data = Vec::new();
        let mut function_evaluations = 0;

        let mut population: Vec<Vec<EvoPheno>> =
            (0..dimensions).map(|_| mapc!(POPULATION_SIZE, |_|
                EvoPheno::new_from_vec(&mapc!(1, |_|
                    thread_rng().gen::<u16>())))).collect();
        let pop2 = population.clone();

        for d in 0..dimensions as usize {
            for individual in &mut population[d] {
                individual.fitness = function.calc(&convert(&mapc!(dimensions as usize,
                |dp| {
                    if dp == d {
                        &individual.val
                    } else {
                        &pop2[dp][thread_rng().gen_range(0, POPULATION_SIZE)].val
                    }
                }), bounds));
                function_evaluations += 1;
            }
        }

        for d in 0..dimensions as usize {
            let rand = thread_rng().gen_range(0, POPULATION_SIZE);
            best_species.push(population[d][rand].val.clone());
        }

        let mut best_found_cost =
            function.calc(&convert(&mapc!(dimensions as usize,
                |i| &best_species[i]), bounds));
        function_evaluations += 1;

        while function_evaluations < MAX_FUNCTION_EVALUATIONS {
            for d in 0..dimensions as usize {
                population[d] = select_new_population(&population[d]);
                population[d] = recombine(&population[d]);

                for i in &mut population[d] {
                    i.clever_mutate(function_evaluations);
                }

                let mut best_individual = 0;
                let mut best_cost = f64::MAX;

                for (i, individual) in population[d].iter_mut().enumerate() {
                    individual.fitness = function.calc(&convert(&mapc!(dimensions as usize,
                    |dp| {
                        if dp == d {
                            &individual.val
                        } else {
                            &best_species[dp]
                        }
                    }), bounds));
                    function_evaluations += 1;

                    if individual.fitness < best_cost {
                        best_cost = individual.fitness;
                        best_individual = i;
                    }
                }

                best_species[d] = population[d][best_individual].val.clone();

                let fitness = function.calc(&convert(&mapc!(dimensions as usize,
                    |i| &best_species[i]), bounds));
                function_evaluations += 1;
                if fitness < best_found_cost {
                    best_found_cost = fitness;
                }
            }
            fitness_data.push(best_found_cost);
        }
        let output: Vec<String> = fitness_data.iter().map(|n| n.to_string()).collect();
        writeln!(ccga4_file, "{}", output.join(",")).unwrap();
    }
}

macro_rules! function_factory {
    ($([$element:ident, $bounds:expr, $dimensions:expr],)*) => (
        enum Function {
            $($element),*
        }

        impl Function {
            fn bounds(&self) -> f64 {
                match *self {
                    $(Function::$element => $bounds),*
                }
            }

            fn dimensions(&self) -> u32 {
                match *self {
                    $(Function::$element => $dimensions),*
                }
            }

            fn calc(&self, vec: &[f64]) -> f64 {
                let n = vec.len();
                let x = vec[0];
                let y = vec[1];

                match *self {
                    Function::Rosenbrock => 100.0 * sq!(sq!(x) - y) + sq!(1.0 - x),
                    Function::Schwefel => 418.982887 * (n as f64) - sum!(n, |i| vec[i] * sin!(sqrt!(abs!(vec[i])))),
                    Function::Rastrigin => 3.0 * (n as f64) + sum!(n, |i| sq!(vec[i]) - 3.0 * cos!(2.0 * PI * vec[i])),
                    Function::Griewangk => 1.0 + sum!(n, |i| sq!(vec[i]) / 4000.0) - product!(n, |i| cos!(vec[i] / sqrt!(((i + 1) as f64)))),
                    Function::Ackley => {
                        20.0 + E - (20.0 * exp!(-0.2 * sqrt!(1.0 / (n as f64) * sum!(n, |i| sq!(vec[i]))))) - exp!(1.0 / (n as f64) * sum!(n, |i| cos!(2.0 * PI * vec[i])))
                    },
                }
            }
        }

        fn main() {
            let args: Vec<String> = env::args().collect();

            if args.len() < 2 {
                eprintln!("Usage: coopco <function>");
                std::process::exit(1);
            }

            let function = match args[1].as_str() {
                $(stringify!($element) => Function::$element,)*
                _ => {
                    println!("Function has not been implemented...\nImplemented:");
                    $(println!("  - {}", stringify!($element));)*
                    std::process::exit(1)
                }
            };

            run_ga(&function);
        }
    )
}

function_factory!(
    [Rosenbrock, 2.048, 2],
    [Griewangk, 600.0, 10],
    [Schwefel, 500.0, 10],
    [Rastrigin, 5.12, 20],
    [Ackley, 30.0, 10],
);
