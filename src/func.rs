use std::f64::consts::{PI, E};

macro_rules! exp { ($a: expr) =>    ($a.exp()) }
macro_rules! sqrt { ($a: expr) =>   ($a.sqrt()) }
macro_rules! sin { ($a: expr) =>    ($a.sin()) }
macro_rules! cos { ($a: expr) =>    ($a.cos()) }
macro_rules! abs { ($a: expr) =>    ($a.abs()) }
macro_rules! sq { ($a: expr) =>     ($a * $a) }
macro_rules! sum { ($a: expr) =>        ($a.fold(0.0, |a, b| a + b)) }
macro_rules! product { ($a: expr) =>    ($a.fold(1.0, |a, b| a * b)) }

pub trait Function {
    fn calc(&self, x: Vec<f64>) -> f64;
}

pub struct Rastrigin;
pub struct Schwefel;
pub struct Griewangk;
pub struct Ackley;
pub struct Rosenbrock;

impl Function for Rastrigin {
    fn calc(&self, x: Vec<f64>) -> f64 {
    	let n = x.len();
    	let sum: f64 = sum!((0 .. n).map(|i| sq!(x[i]) - 3.0 * cos!(2.0 * PI * x[i])));

        3.0 * (n as f64) + sum
    }
}

impl Function for Schwefel {
    fn calc(&self, x: Vec<f64>) -> f64 {
    	let n = x.len();
    	let sum: f64 = sum!((0 .. n).map(|i| x[i] * sin!(sqrt!(abs!(x[i])))));

    	418.982887 * (n as f64) - sum
    }
}

impl Function for Griewangk {
    fn calc(&self, x: Vec<f64>) -> f64 {
    	let n = x.len();
    	let sum: f64 = sum!((0 .. n).map(|i| sq!(x[i]) / 4000.0));
    	let product: f64 = product!((0 .. n).map(|i| cos!(x[i] / sqrt!(((i + 1) as f64)))));

    	1.0 + sum - product
    }
}

impl Function for Ackley {
    fn calc(&self, x: Vec<f64>) -> f64 {
    	let n = x.len();
    	let sum1: f64 = sum!((0 .. n).map(|i| sq!(x[i])));
    	let sum2: f64 = sum!((0 .. n).map(|i| cos!(2.0 * PI * x[i])));

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

#[cfg(test)]
mod tests {
	use func::*;

    #[test]
    fn it_works() {
    	assert_approx_eq!(0.0, Rastrigin.calc([0.0; 20].to_vec()));
        assert_approx_eq!(0.0, Schwefel.calc([420.968746; 10].to_vec()), 1e-5f64);
        assert_approx_eq!(0.0, Griewangk.calc([0.0; 10].to_vec()));
        assert_approx_eq!(0.0, Ackley.calc([0.0; 30].to_vec()));
        assert_approx_eq!(0.0, Rosenbrock.calc([1.0; 2].to_vec()));
    }
}