#![feature(alloc_system)] extern crate alloc_system;
#[cfg(test)] #[macro_use] extern crate assert_approx_eq;
extern crate bit_vec;
extern crate rand;

use bit_vec::BitVec;
use rand::{Rng, weak_rng};
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
const ITERATIONS: u32 = 1000;

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

    fn crossover(&self, other: &EvoPheno) -> EvoPheno {
        let mut new = self.val.clone();
        let mut rand = weak_rng();

        for i in 0..self.val.len() {
            if rand.next_f64() < 0.6 {
                new.set(i, other.val[i]);
            }
        }

        EvoPheno::new(new)
    }

    fn multi_crossover(&self, other: &EvoPheno) -> EvoPheno {
        let mut new = self.val.clone();
        let len = self.val.len();
        let (low, high) = {
            let a = weak_rng().gen_range(0, len);
            let b = weak_rng().gen_range(0, len);

            if a < b {
                (a, b)
            } else {
                (b, a)
            }
        };

        for i in 0..self.val.len() {
            if i > low && i < high {
                new.set(i, other.val[i]);
            }
        }

        EvoPheno::new(new)
    }

    fn mutate(&self) -> EvoPheno {
        let mut new = self.val.clone();
        let mut rand = weak_rng();

        for i in 0..self.val.len() {
            if rand.next_f64() < (1f64 / self.val.len() as f64) {
                new.set(i, !self.val[i]);
            }
        }

        EvoPheno::new(new)
   }

    fn clever_mutate(&self, iteration: u32) -> EvoPheno {
        let mut new = self.val.clone();
        let mut rand = weak_rng();
        let len = self.val.len();

        for i in 0..len {
            let prob = 0.1 * (cos!(PI * (i / len) as f64 - PI * (iteration / ITERATIONS) as f64)) + 0.7;

            if rand.next_f64() < prob {
                new.set(i, !self.val[i]);
            }
        }

        EvoPheno::new(new)
    }
}

fn tournament(population: &[EvoPheno], offset: usize) -> (usize, usize) {
    let pop_slice = &population[offset..offset + POPULATION_SIZE];

    let a_index = weak_rng().gen_range(0, POPULATION_SIZE) as usize;
    let b_index = weak_rng().gen_range(0, POPULATION_SIZE) as usize;

    if pop_slice[a_index].fitness < pop_slice[b_index].fitness {
        (a_index, b_index)
    } else {
        (b_index, a_index)
    }
}

fn fitness_selection(population: &[EvoPheno], offset: usize) -> usize {
    let mut s = population[offset..offset + POPULATION_SIZE].to_vec();
    let total_fitness = sum!(s.len(), |i| s[i].fitness);
    s.sort_by(|a: &EvoPheno, b: &EvoPheno| a.fitness.partial_cmp(&b.fitness).unwrap());
    s.reverse();

    let random_number = weak_rng().gen_range(0.0, total_fitness);
    let mut sum = 0.0;

    for (i, item) in s.iter().enumerate() {
        sum += item.fitness;

        if sum > random_number {
            return i;
        }
    }

    unreachable!();
}

fn real_val(bounds: f64, value: u16) -> f64 {
    -bounds + (f64::from(value) / f64::from(u16::MAX) * bounds * 2.0)
}

fn make_vec_species(population: &[EvoPheno], child_val: &SmBitVec, child_pos: usize, bounds: f64, top: bool) -> Vec<f64> {
    let mut rand = weak_rng();
    convert(&mapc!(population.len() / POPULATION_SIZE, |d| {
        if d == child_pos {
            child_val
        } else {
            let offset = (d as usize) * POPULATION_SIZE;
            if top {
                &get_best(&population[offset..offset + POPULATION_SIZE]).val
            } else {
                &population[rand.gen_range(offset, offset + POPULATION_SIZE)].val
            }
        }
    }), bounds)
}

fn make_population(size: usize, function: &Function, dimensions: u32) -> Vec<EvoPheno> {
    let bounds = function.bounds();
    let func_dimensions = function.dimensions();
    let mut rand = weak_rng();

    let mut population: Vec<EvoPheno> = mapc!(size, |_|
        EvoPheno::new_from_vec(&mapc!(dimensions, |_| rand.gen::<u16>()))
    );

    let pop2 = population.clone();

    for p in &mut population {
        p.fitness = if dimensions == func_dimensions {
            function.calc(&make_vec(&p.val, bounds))
        } else {
            function.calc(&convert(&mapc!(func_dimensions, |d| {
                let offset = (d as usize) * POPULATION_SIZE;
                &pop2[rand.gen_range(offset, offset + POPULATION_SIZE)].val
            }), bounds))
        };
    }

    population
}

fn get_best(population: &[EvoPheno]) -> &EvoPheno {
    population
        .iter()
        .min_by(|a: &&EvoPheno, b: &&EvoPheno| a.fitness.partial_cmp(&b.fitness).unwrap())
        .unwrap()
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
    let function_min = function.known_min();
    let bounds = function.bounds();

    let mut ga_file = File::create("ga.csv").unwrap();
    let mut ccga1_file = File::create("ccga1.csv").unwrap();
    let mut ccga1_clever_file = File::create("ccga1_clever.csv").unwrap();

    println!("GA");
    for _ in 0..50 {
        let mut population = make_population(POPULATION_SIZE, function, dimensions);

        let fitness_data: Vec<f64> = mapc!(ITERATIONS, |iterations| {
            let parent1_index = fitness_selection(&population, 0);
            let parent2_index = fitness_selection(&population, 0);
            //let parent1_index = tournament(&population, 0).0;
            //let parent2_index = tournament(&population, 0).0;

            let mut child = population[parent1_index].multi_crossover(&population[parent2_index]).mutate();
            child.fitness = function.calc(&make_vec(&child.val, bounds));

            let new_index = tournament(&population, 0).1;
            //let new_index = fitness_selection(&population, 0);
            std::mem::replace(&mut population[new_index], child);

            (get_best(&population).fitness - function_min).abs()
        });
        let output: Vec<String> = fitness_data.iter().map(|n| n.to_string()).collect();
        writeln!(ga_file, "{}", output.join(",")).unwrap();
    }

    println!("CCGA1");
    for _ in 0..50 {
        let mut population = make_population(POPULATION_SIZE * (dimensions as usize), function, 1);

        let fitness_data: Vec<f64> = mapc!(ITERATIONS, |iterations| {
            for d in 0..dimensions {
                let offset = (d as usize) * POPULATION_SIZE;
                //let parent1_index = offset + tournament(&population, offset).0;
                //let parent2_index = offset + tournament(&population, offset).0;
                let parent1_index = offset + fitness_selection(&population, offset);
                let parent2_index = offset + fitness_selection(&population, offset);

                let mut child = population[parent1_index].multi_crossover(&population[parent2_index]).mutate();
                child.fitness = function.calc(&make_vec_species(&population, &child.val, d as usize, bounds, true));

                //let new_index = offset + tournament(&population, offset).1;
                let new_index = offset + fitness_selection(&population, offset);
                std::mem::replace(&mut population[new_index], child);
            }

            (function.calc(&convert(&mapc!(dimensions, |d| {
                let offset = (d as usize) * POPULATION_SIZE;
                &get_best(&population[offset..offset + POPULATION_SIZE]).val
            }), bounds)) - function_min).abs()
        });
        let output: Vec<String> = fitness_data.iter().map(|n| n.to_string()).collect();
        writeln!(ccga1_file, "{}", output.join(",")).unwrap();
    }

    println!("CCGA1 Clever");
    for _ in 0..50 {
        let mut population = make_population(POPULATION_SIZE * (dimensions as usize), function, 1);

        let fitness_data: Vec<f64> = mapc!(ITERATIONS, |iterations| {
            for d in 0..dimensions {
                let offset = (d as usize) * POPULATION_SIZE;
                //let parent1_index = offset + tournament(&population, offset).0;
                //let parent2_index = offset + tournament(&population, offset).0;
                let parent1_index = offset + fitness_selection(&population, offset);
                let parent2_index = offset + fitness_selection(&population, offset);

                let mut child = population[parent1_index].multi_crossover(&population[parent2_index]).clever_mutate(iterations);
                child.fitness = function.calc(&make_vec_species(&population, &child.val, d as usize, bounds, true));

                //let new_index = offset + tournament(&population, offset).1;
                let new_index = offset + fitness_selection(&population, offset);
                std::mem::replace(&mut population[new_index], child);
            }

            (function.calc(&convert(&mapc!(dimensions, |d| {
                let offset = (d as usize) * POPULATION_SIZE;
                &get_best(&population[offset..offset + POPULATION_SIZE]).val
            }), bounds)) - function_min).abs()
        });
        let output: Vec<String> = fitness_data.iter().map(|n| n.to_string()).collect();
        writeln!(ccga1_clever_file, "{}", output.join(",")).unwrap();
    }

    /*for _ in 0..50 {
        let mut population = make_population(POPULATION_SIZE * (dimensions as usize), function, 1);

        let fitness_data: Vec<(f64, f64)> = mapc!(ITERATIONS, |iterations| {
            for d in 0..dimensions {
                let offset = (d as usize) * POPULATION_SIZE;
                let parent1_index = offset + tournament(&population, offset).0;
                let parent2_index = offset + tournament(&population, offset).0;

                let mut child = population[parent1_index].multi_crossover(&population[parent2_index]).mutate();
                let top_best_fitness = function.calc(&make_vec_species(&population, &child.val, d as usize, bounds, true));
                let random_fitness = function.calc(&make_vec_species(&population, &child.val, d as usize, bounds, false));
                child.fitness = top_best_fitness.min(random_fitness);

                let new_index = offset + tournament(&population, offset).1;
                std::mem::replace(&mut population[new_index], child);
            }

           (f64::from(iterations), (function.calc(&convert(&mapc!(dimensions, |d| {
                let offset = (d as usize) * POPULATION_SIZE;
                &get_best(&population[offset..offset + POPULATION_SIZE]).val
            }), bounds)) - function_min).abs())

        });
        p_fitness_ccga2.lines("", fitness_data).line_width(1);
    }*/
}

macro_rules! function_factory {
    ($([$element:ident, $bounds:expr, $dimensions:expr, $knownmin:expr],)*) => (
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

            fn known_min(&self) -> f64 {
                match *self {
                    $(Function::$element => $knownmin),*
                }
            }

            fn calc(&self, vec: &[f64]) -> f64 {
                let n = vec.len();
                let x = vec[0];
                let y = vec[1];

                match *self {
                    Function::Sphere => sum!(n, |i| sq!(vec[i])),
                    Function::Matyas => 0.26 * (sq!(x) + sq!(y)) - 0.48 * x * y,
                    Function::Rosenbrock => 100.0 * sq!(sq!(x) - y) + sq!(1.0 - x),
                    Function::Booth => sq!(x + 2.0 * y - 7.0) + sq!(2.0 * x + y - 5.0),
                    Function::Easom => - cos!(x) * cos!(y) * exp!(-(sq!(x - PI) + sq!(y - PI))),
                    Function::Schwefel => 418.982887 * (n as f64) - sum!(n, |i| vec[i] * sin!(sqrt!(abs!(vec[i])))),
                    Function::StyblinskiTang => sum!(n, |i| vec[i].powi(4) - 16.0 * sq!(vec[i]) + 5.0 * vec[i]) / 2.0,
                    Function::ThreeHump => 2.0 * sq!(x) - 1.05 * x.powi(4) + (x.powi(6) / 6.0) + (x * y) + sq!(y),
                    Function::Beale => sq!(1.5 - x + x * y) + sq!(2.25 - x + x * sq!(y)) + sq!(2.625 - x + x * y.powi(3)),
                    Function::HolderTable => - abs!(sin!(x) * cos!(y) * exp!(abs!(1.0 - (sqrt!(sq!(x) + sq!(y)) / PI)))),
                    Function::Schaffer4 => 0.5 + ((sq!(cos!(sin!(abs!(sq!(x) - sq!(y))))) - 0.5) / sq!(1.0 + 0.001 * (sq!(x + sq!(y))))),
                    Function::Schaffer2 => 0.5 + ((sq!(sin!(sq!(x) - sq!(y))) - 0.5) / sq!(1.0 + 0.001 * (sq!(x + sq!(y))))),
                    Function::Rastrigin => 3.0 * (n as f64) + sum!(n, |i| sq!(vec[i]) - 3.0 * cos!(2.0 * PI * vec[i])),
                    Function::Eggholder => - (y + 47.0) * sin!(sqrt!(abs!((x / 2.0) + y + 47.0))) - x * sin!(sqrt!(abs!(x - (y + 47.0)))),
                    Function::CrossInTray => - 0.0001 * (abs!(sin!(x) * sin!(y) * exp!(abs!(100.0 - sqrt!(sq!(x) + sq!(y)) / PI))) + 1.0).powf(0.1),
                    Function::Griewangk => 1.0 + sum!(n, |i| sq!(vec[i]) / 4000.0) - product!(n, |i| cos!(vec[i] / sqrt!(((i + 1) as f64)))),
                    Function::Levi13 => sq!(sin!(3.0 * PI * x)) + sq!(x - 1.0) * (1.0 + sq!(sin!(3.0 * PI * y))) + sq!(y - 1.0) * (1.0 + sq!(sin!(2.0 * PI * y))),
                    Function::Ackley => {
                        20.0 + E - (20.0 * exp!(-0.2 * sqrt!(1.0 / (n as f64) * sum!(n, |i| sq!(vec[i]))))) - exp!(1.0 / (n as f64) * sum!(n, |i| cos!(2.0 * PI * vec[i])))
                    },
                    Function::GoldsteinPrice => {
                        (1.0 + sq!(x + y + 1.0) * (19.0 - 14.0*x + 3.0*sq!(x) - 14.0*y + 6.0*x*y + 3.0*sq!(y)))
                        *
                        (30.0 + sq!(2.0*x - 3.0*y) * (18.0 - 32.0*x + 12.0*sq!(x) + 48.0*y - 36.0*x*y + 27.0*sq!(y)))
                    }
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
    [Rosenbrock, 2.048, 2, 0.0],
    [Griewangk, 600.0, 10, 0.0],
    [Schwefel, 500.0, 10, 0.0],
    [Rastrigin, 5.12, 20, 0.0],
    [Ackley, 30.0, 10, 0.0],

    [Beale, 4.5, 2, 0.0],
    [ThreeHump, 5.0, 2, 0.0],
    [Matyas, 10.0, 2, 0.0],
    [Levi13, 10.0, 2, 0.0],
    [Booth, 10.0, 2, 0.0],
    [CrossInTray, 10.0, 2, -2.06261],
    [HolderTable, 10.0, 2, -19.2085],
    [Schaffer2, 100.0, 2, 0.0],
    [Schaffer4, 100.0, 2, 0.292579],
    [Easom, 100.0, 2, -1.0],
    [Sphere, 100.0, 2, 0.0],
    [Eggholder, 512.0, 2, -959.6407],
    [StyblinskiTang, 5.0, 2, -2.903534], // An odd one
    [GoldsteinPrice, 2.0, 2, 3.0],
);

#[cfg(test)]
mod tests {
    use super::*;
    use super::Function::*;
    use std::f64::consts::PI;

    #[test]
    fn it_works() {
        assert_approx_eq!(0.0, Rastrigin.calc(&[0.0; 20]));
        assert_approx_eq!(0.0, Schwefel.calc(&[420.968746; 10]), 1e-5f64);
        assert_approx_eq!(0.0, Griewangk.calc(&[0.0; 10]));
        assert_approx_eq!(0.0, Ackley.calc(&[0.0; 30]));
        assert_approx_eq!(0.0, Rosenbrock.calc(&[1.0; 2]));

        assert_approx_eq!(0.0, Sphere.calc(&[0.0; 2]));
        assert_approx_eq!(-1.0, Easom.calc(&[PI; 2]));
        assert_approx_eq!(0.0, ThreeHump.calc(&[0.0; 2]));
        assert_approx_eq!(0.0, Matyas.calc(&[0.0; 2]));
        assert_approx_eq!(0.0, Schaffer2.calc(&[0.0; 2]));
        assert_approx_eq!(0.292579, Schaffer4.calc(&[0.0, 1.25313]), 1e-3f64);
        assert_approx_eq!(0.0, Levi13.calc(&[1.0; 2]));

        assert_approx_eq!(-2.06261, CrossInTray.calc(&[1.34941, 1.34941]), 1e-5f64);
        assert_approx_eq!(-2.06261, CrossInTray.calc(&[-1.34941, 1.34941]), 1e-5f64);
        assert_approx_eq!(-2.06261, CrossInTray.calc(&[1.34941, -1.34941]), 1e-5f64);
        assert_approx_eq!(-2.06261, CrossInTray.calc(&[-1.34941, -1.34941]), 1e-5f64);

        assert_approx_eq!(0.0, Booth.calc(&[1.0, 3.0]));
        assert_approx_eq!(-959.6407, Eggholder.calc(&[512.0, 404.2319]), 1e-4f64);
        assert_approx_eq!(0.0, Beale.calc(&[3.0, 0.5]));

        assert_approx_eq!(-19.2085, HolderTable.calc(&[8.05502, 9.66459]), 1e-5f64);
        assert_approx_eq!(-19.2085, HolderTable.calc(&[-8.05502, 9.66459]), 1e-5f64);
        assert_approx_eq!(-19.2085, HolderTable.calc(&[8.05502, -9.66459]), 1e-5f64);
        assert_approx_eq!(-19.2085, HolderTable.calc(&[-8.05502, -9.66459]), 1e-5f64);

        assert_approx_eq!(3.0, GoldsteinPrice.calc(&[0.0, -1.0]));

        assert_approx_eq!(-2.0, real_val(5.0, u16::MAX/10*3), 1e-3f64);
    }
}
