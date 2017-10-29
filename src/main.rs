#![feature(iterator_step_by)]
#![feature(alloc_system)] extern crate alloc_system;
#[macro_use] extern crate assert_approx_eq;
extern crate rand;

mod func;

fn main() {
    
}

/*use rand::{thread_rng, Rng};

const TARGET_LEN: usize = 16;

#[derive(Clone, Debug)]
struct EvoPheno {
    text: Vec<u8>,
    fitness: i32
}

impl EvoPheno {
    fn new(t: Vec<u8>) -> EvoPheno {
        let mut fitness: i32 = 0;
        let target_chars = TARGET.as_bytes();

        for i in 0..TARGET_LEN {
            if target_chars[i] == t[i] {
                fitness += 1;
            }
        }

        EvoPheno {
            text: t,
            fitness: fitness
        }
    }

    fn new_random() -> EvoPheno {
        EvoPheno::new((0..TARGET_LEN).map(|_| thread_rng().gen_range(32, 127)).collect())
    }

    fn crossover(&self, other: &EvoPheno) -> EvoPheno {
        EvoPheno::new((0..TARGET_LEN).map(|i| if rand::random::<f64>() < 0.6 {
            self.text[i]
        } else {
            other.text[i]
        }).collect())
    }

    fn mutate(&self) -> EvoPheno {
        EvoPheno::new((0..TARGET_LEN).map(|i| if rand::random::<f64>() < (1f64 / TARGET_LEN as f64) {
            thread_rng().gen_range(32, 127)
        } else {
            self.text[i]
        }).collect())
    }
}

fn main() {
    let mut population: Vec<EvoPheno> = (0..population_size).map(|_| EvoPheno::new_random()).collect();

    let mut iterations = 0;
    let mut gen = 0;

    loop {
        gen += 1;
        iterations += 1;

        // select Pop(gen) from Pop(gen - 1) based upon fitness
        // apply genetic operators to Pop(gen)
        // evaluate fitness of each individual in Pop(gen)
    }
}

*/