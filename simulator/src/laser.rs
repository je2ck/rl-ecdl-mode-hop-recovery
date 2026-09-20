use pyo3::exceptions::PyValueError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use pyo3::prelude::*;

use eyre;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, thread_rng};
use rand_distr::{Distribution, Normal};

pub const MEAN_PZT_UP_CHANGE: f64 = 0.0002528756410228174;
pub const STD_PZT_UP_CHANGE: f64 = 9.808018129040478e-06;
pub const MEAN_PZT_DOWN_CHANGE: f64 = -0.0002405155555552916;
pub const STD_PZT_DOWN_CHANGE: f64 = 6.081280878654801e-06;

/// Returns the path to the simulator data directory.
/// Reads from the `LASER_SIM_DATA_DIR` environment variable if set,
/// otherwise falls back to `<CARGO_MANIFEST_DIR>/data`.
pub fn data_dir() -> PathBuf {
    match std::env::var("LASER_SIM_DATA_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"),
    }
}

#[derive(Debug)]
pub struct DiodeLaser {
    stable_states: HashMap<Signal, Vec<LaserState>>,
    unstable_states: Vec<LaserState>,

    // internal states
    pub internal_state: LaserInternalState,
    pub current_range: u32,
}

#[derive(Debug, Clone)]
pub struct LaserInternalState {
    pub latest_state: LaserState,
    pzt_moves: f64, // move by pzt control
}

impl Default for LaserInternalState {
    fn default() -> Self {
        LaserInternalState {
            latest_state: LaserState::default(),
            pzt_moves: 0.0,
        }
    }
}

#[pyclass]
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LaserState {
    #[pyo3(get, set)]
    pub current: f64,
    #[pyo3(get, set)]
    pub voltage: f64,
    #[pyo3(get, set)]
    pub frequency: f64,
    #[pyo3(get, set)]
    pub stable: bool,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Signal {
    #[pyo3(name = "CurrentUp")]
    CurrentUp,
    #[pyo3(name = "CurrentUpInStable")]
    CurrentUpInStable,
    #[pyo3(name = "CurrentUpInUnstable")]
    CurrentUpInUnstable,
    #[pyo3(name = "CurrentDown")]
    CurrentDown,
    #[pyo3(name = "PZTUp")]
    PZTUp,
    #[pyo3(name = "PZTDown")]
    PZTDown,
}

#[pymethods]
impl Signal {
    #[staticmethod]
    pub fn from_str(value: &str) -> PyResult<Self> {
        match value {
            "CurrentUp" => Ok(Signal::CurrentUp),
            "CurrentUpInStable" => Ok(Signal::CurrentUpInStable),
            "CurrentUpInUnstable" => Ok(Signal::CurrentUpInUnstable),
            "CurrentDown" => Ok(Signal::CurrentDown),
            "PZTUp" => Ok(Signal::PZTUp),
            "PZTDown" => Ok(Signal::PZTDown),
            _ => Err(PyValueError::new_err("Invalid signal")),
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            Signal::CurrentUp => "CurrentUp",
            Signal::CurrentUpInStable => "CurrentUpInStable",
            Signal::CurrentUpInUnstable => "CurrentUpInUnstable",
            Signal::CurrentDown => "CurrentDown",
            Signal::PZTUp => "PZTUp",
            Signal::PZTDown => "PZTDown",
        }
    }
}

impl DiodeLaser {
    pub fn new(current_range: u32) -> eyre::Result<Self> {
        let abs_path = data_dir();
        let states = read_from_file(abs_path.join("current_up_stable.json"))?;
        let mut stable_states = HashMap::new();
        stable_states
            .entry(Signal::CurrentUpInStable)
            .or_insert(states);

        let states = read_from_file(abs_path.join("current_up_unstable.json"))?;
        stable_states
            .entry(Signal::CurrentUpInUnstable)
            .or_insert(states);

        let mut states = read_from_file(abs_path.join("current_down_minus.json"))?;
        states.retain(|state| state.stable);
        stable_states.entry(Signal::CurrentDown).or_insert(states);

        let mut unstable_states = read_from_file(abs_path.join("current_all.json"))?;
        unstable_states.retain(|state| !state.stable);

        Ok(DiodeLaser {
            stable_states,
            unstable_states,
            internal_state: LaserInternalState::default(),
            current_range,
        })
    }

    pub fn reset(&mut self, state: LaserState) {
        self.internal_state = LaserInternalState::default();
        self.with_laser_state(state);
    }

    pub fn with_laser_state(&mut self, state: LaserState) {
        self.internal_state.latest_state = state;
    }

    pub fn emit(&mut self, signal: Signal) -> (LaserState, f64) {
        let pzt_amount = match signal {
            Signal::PZTDown => pzt_down_sample(),
            Signal::PZTUp => pzt_up_sample(),
            _ => 0.0,
        };
        let n = 15usize;
        match signal {
            Signal::CurrentUp => {
                for _ in 0..n {
                    self.internal_state.latest_state = self.get_state(&signal, pzt_amount);
                }
                for _ in 0..(n - 1) {
                    self.internal_state.latest_state =
                        self.get_state(&Signal::CurrentDown, pzt_amount);
                }
            }
            _ => self.internal_state.latest_state = self.get_state(&signal, pzt_amount),
        };
        (self.internal_state.latest_state.clone(), pzt_amount)
    }

    pub fn move_horizontal(&mut self, amount: f64) {
        self.stable_states.iter_mut().for_each(|(_, states)| {
            states.iter_mut().for_each(|state| {
                state.current += amount;
            });
        });
        self.unstable_states.iter_mut().for_each(|state| {
            state.current += amount;
        });
    }

    pub fn move_vertical(&mut self, amount: f64) {
        self.stable_states.iter_mut().for_each(|(_, states)| {
            states.iter_mut().for_each(|state| {
                state.frequency += amount;
            });
        });
        self.unstable_states.iter_mut().for_each(|state| {
            state.frequency += amount;
        });
    }

    fn get_state(&mut self, signal: &Signal, pzt_amount: f64) -> LaserState {
        let state = match (self.internal_state.latest_state.stable, signal.clone()) {
            (true, Signal::CurrentUp) => {
                let new_current =
                    round_to_decimal_places(self.internal_state.latest_state.current + 0.1, 1);
                let mut states = Vec::new();
                states.extend(self.stable_states.get(&Signal::CurrentUpInStable).unwrap());
                states.extend(
                    self.stable_states
                        .get(&Signal::CurrentUpInUnstable)
                        .unwrap(),
                );

                let matching_states: Vec<_> = states
                    .into_iter()
                    .filter(|state| round_to_decimal_places(state.current, 1) == new_current)
                    .cloned()
                    .collect();

                if let Some(state) = matching_states.choose(&mut thread_rng()) {
                    let next_frequency = add_random_3decimals(state.frequency);
                    let mut state = state.clone();
                    state.frequency = next_frequency;
                    state
                } else {
                    let mut state = self.internal_state.latest_state.clone();
                    state.current = new_current;
                    state.stable = false;
                    state.frequency = add_random_4decimal(state.frequency);
                    state
                }
            }
            (false, Signal::CurrentUp) => {
                let mut rng = thread_rng();
                let mut state = self.unstable_states.choose(&mut rng).unwrap().clone();
                state.current =
                    round_to_decimal_places(self.internal_state.latest_state.current + 0.1, 1);
                state.voltage = self.internal_state.latest_state.voltage;

                let mut stable_states = Vec::new();
                stable_states.extend(
                    self.stable_states
                        .get(&Signal::CurrentUpInUnstable)
                        .unwrap(),
                );
                let current = self.internal_state.latest_state.current;
                let min_value = stable_states
                    .iter()
                    .filter(|state| state.current > current)
                    .map(|state| state.current)
                    .min_by(|a, b| a.partial_cmp(b).unwrap());

                let next_closests: Vec<_> = stable_states
                    .iter()
                    .filter(|state| {
                        min_value.is_some()
                            && round_to_decimal_places(state.current, 1)
                                == round_to_decimal_places(min_value.unwrap(), 1)
                    })
                    .collect();

                if let Some(next_closest) = next_closests.choose(&mut thread_rng()) {
                    let steps_to_next =
                        (next_closest.current - self.internal_state.latest_state.current).abs()
                            / 0.1;
                    if steps_to_next as usize <= 2 {
                        state.stable = true;
                        state.frequency = add_random_3decimals(next_closest.frequency);
                    } else {
                        state.frequency = add_random_4decimal(next_closest.frequency);
                    }
                    return state;
                }

                stable_states.extend(self.stable_states.get(&Signal::CurrentUpInStable).unwrap());
                stable_states.extend(self.stable_states.get(&Signal::CurrentDown).unwrap());
                let min_value = stable_states
                    .iter()
                    .filter(|state| state.current > current)
                    .map(|state| state.current)
                    .min_by(|a, b| a.partial_cmp(b).unwrap());

                let next_closests: Vec<_> = stable_states
                    .iter()
                    .filter(|state| {
                        min_value.is_some()
                            && round_to_decimal_places(state.current, 1)
                                == round_to_decimal_places(min_value.unwrap(), 1)
                    })
                    .collect();

                if let Some(next_closest) = next_closests.choose(&mut thread_rng()) {
                    let steps_to_next =
                        (next_closest.current - self.internal_state.latest_state.current).abs()
                            / 0.1;
                    if steps_to_next as usize <= 2 {
                        state.stable = true;
                        state.frequency = add_random_3decimals(next_closest.frequency);
                    } else {
                        state.frequency = add_random_4decimal(next_closest.frequency);
                    }
                }
                state
            }
            (true, Signal::CurrentDown) => {
                let new_current =
                    round_to_decimal_places(self.internal_state.latest_state.current - 0.1, 1);
                let mut states = Vec::new();
                states.extend(self.stable_states.get(&Signal::CurrentDown).unwrap());

                let matching_states: Vec<_> = states
                    .into_iter()
                    .filter(|state| round_to_decimal_places(state.current, 1) == new_current)
                    .cloned()
                    .collect();

                if let Some(state) = matching_states.choose(&mut thread_rng()) {
                    let next_frequency = add_random_3decimals(state.frequency);
                    let mut state = state.clone();
                    state.frequency = next_frequency;
                    state
                } else {
                    let mut state = self.internal_state.latest_state.clone();
                    state.current = new_current;
                    state.stable = false;
                    state.frequency = add_random_4decimal(state.frequency);
                    state
                }
            }
            (false, Signal::CurrentDown) => {
                let new_current =
                    round_to_decimal_places(self.internal_state.latest_state.current - 0.1, 1);
                let mut rng = thread_rng();
                let mut state = self.unstable_states.choose(&mut rng).unwrap().clone();
                state.current = new_current;
                state.voltage = self.internal_state.latest_state.voltage;

                let mut stable_states = Vec::new();
                stable_states.extend(self.stable_states.get(&Signal::CurrentDown).unwrap());
                let current = self.internal_state.latest_state.current;
                let max_value = stable_states
                    .iter()
                    .filter(|state| state.current < current)
                    .map(|state| state.current)
                    .max_by(|a, b| a.partial_cmp(b).unwrap());

                let next_closests: Vec<_> = stable_states
                    .iter()
                    .filter(|state| {
                        max_value.is_some()
                            && round_to_decimal_places(state.current, 1)
                                == round_to_decimal_places(max_value.unwrap(), 1)
                    })
                    .collect();

                if let Some(next_closest) = next_closests.choose(&mut thread_rng()) {
                    let steps_to_next =
                        (next_closest.current - self.internal_state.latest_state.current).abs()
                            / 0.1;
                    if steps_to_next as usize <= 2 {
                        state.stable = true;
                        state.frequency = add_random_3decimals(next_closest.frequency);
                    } else {
                        state.frequency = add_random_4decimal(next_closest.frequency);
                    }
                }
                state
            }
            (_, Signal::PZTUp) => {
                self.internal_state.pzt_moves += 1.0;
                let mut state = self.internal_state.latest_state.clone();
                state.frequency = round_to_decimal_places(state.frequency + pzt_amount, 8);
                state.voltage = round_to_decimal_places(state.voltage + 0.1, 1);
                self.move_vertical(pzt_amount);
                state
            }
            (_, Signal::PZTDown) => {
                self.internal_state.pzt_moves -= 1.0;
                let mut state = self.internal_state.latest_state.clone();
                state.frequency = round_to_decimal_places(state.frequency + pzt_amount, 8);
                state.voltage = round_to_decimal_places(state.voltage - 0.1, 1);
                self.move_vertical(pzt_amount);
                state
            }
            _ => unimplemented!(),
        };
        state
    }

    pub fn choose_random_state_with_seed(
        &self,
        seed: u64,
        lower: f64,
        upper: f64,
        target_frequency: f64,
    ) -> LaserState {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut states = Vec::new();
        states.extend(self.stable_states.get(&Signal::CurrentDown).unwrap());
        states.extend(self.unstable_states.iter());
        let mut states = states
            .into_iter()
            .filter(|state| {
                state.current >= lower
                    && state.current <= upper
                    && round_to_decimal_places(state.frequency, 2)
                        != round_to_decimal_places(target_frequency, 2)
            })
            .collect::<Vec<_>>();
        assert!(!states.is_empty());
        states.shuffle(&mut rng);
        states.pop().unwrap().clone()
    }

    pub fn choose_state_with_current(&self, current: f64) -> LaserState {
        let mut states = Vec::new();
        states.extend(self.stable_states.get(&Signal::CurrentDown).unwrap());
        states.extend(self.unstable_states.iter());
        let matching_states: Vec<_> = states
            .into_iter()
            .filter(|state| round_to_decimal_places(state.current, 1) == current)
            .cloned()
            .collect();
        matching_states[0].clone()
    }

    pub fn choose_edge_state(&self, lower: f64, upper: f64) -> LaserState {
        let mut rng = thread_rng();
        let mut states = Vec::new();
        states.extend(self.stable_states.get(&Signal::CurrentDown).unwrap());
        states.extend(self.unstable_states.iter());
        let mut states = states
            .into_iter()
            .filter(|state| state.current >= lower && state.current <= upper)
            .collect::<Vec<_>>();
        states.sort_by(|a, b| a.current.partial_cmp(&b.current).unwrap());
        assert!(!states.is_empty());

        let mut candid = vec![];
        let n = (8. / 100. * self.current_range as f64) as usize;
        let left_nth = states[n].clone();
        let right_nth = states[states.len() - n - 1].clone();
        candid.push(left_nth);
        candid.push(right_nth);

        let n = (16. / 100. * self.current_range as f64) as usize;
        let left_nth = states[n].clone();
        let right_nth = states[states.len() - n - 1].clone();
        candid.push(left_nth);
        candid.push(right_nth);

        candid.choose(&mut rng).unwrap().clone()
    }

    pub fn choose_random_state(
        &self,
        lower: f64,
        upper: f64,
        target_frequency: f64,
        step_num: usize,
    ) -> LaserState {
        let mut rng = thread_rng();
        let mut states = Vec::new();
        states.extend(self.stable_states.get(&Signal::CurrentDown).unwrap());
        states.extend(self.unstable_states.iter());
        let mut states = states
            .into_iter()
            .filter(|state| {
                state.current >= lower
                    && state.current <= upper
                    && round_to_decimal_places(state.frequency, 2)
                        != round_to_decimal_places(target_frequency, 2)
            })
            .collect::<Vec<_>>();
        states.sort_by(|a, b| a.current.partial_cmp(&b.current).unwrap());
        assert!(!states.is_empty());
        let step = states.len() / step_num;
        let mut candid = (0..step_num)
            .map(|i| states[i * step].clone())
            .collect::<Vec<_>>();
        candid.shuffle(&mut rng);
        candid.pop().unwrap().clone()
    }
}

fn pzt_up_sample() -> f64 {
    let normal = Normal::new(MEAN_PZT_UP_CHANGE, STD_PZT_UP_CHANGE)
        .expect("Invalid distribution parameters");
    let mut rng = thread_rng();
    normal.sample(&mut rng)
}

fn pzt_down_sample() -> f64 {
    let normal = Normal::new(MEAN_PZT_DOWN_CHANGE, STD_PZT_DOWN_CHANGE)
        .expect("Invalid distribution parameters");
    let mut rng = thread_rng();
    normal.sample(&mut rng)
}

pub fn round_to_decimal_places(num: f64, places: i32) -> f64 {
    let factor = 10f64.powi(places);
    (num * factor).round() / factor
}

fn add_random_3decimals(frequency: f64) -> f64 {
    let mut rng = thread_rng();
    let dist = Uniform::new_inclusive(100, 999);
    let random_part: f64 = rng.sample(dist) as f64 / 100_000_000.0;
    (frequency * 100_000.0).round() / 100_000.0 + random_part
}

fn add_random_4decimal(frequency: f64) -> f64 {
    let mut rng = thread_rng();
    let dist = Uniform::new_inclusive(1000, 9999);
    let random_part: f64 = rng.sample(dist) as f64 / 100_000_000.0;
    (frequency * 10_000.0).round() / 10_000.0 + random_part
}

pub fn read_from_file<P: AsRef<Path>>(file_path: P) -> eyre::Result<Vec<LaserState>> {
    let file = File::open(file_path.as_ref())?;
    let reader = BufReader::new(file);
    let mut states = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let state: LaserState = serde_json::from_str(&line).expect("Failed to parse JSON line");
        states.push(state);
    }
    states.iter_mut().for_each(|state| {
        state.current = round_to_decimal_places(state.current, 1);
        state.voltage = round_to_decimal_places(state.voltage, 1);
        state.frequency = round_to_decimal_places(state.frequency, 5);
    });
    states.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap());
    states.dedup_by(|a, b| a.frequency == b.frequency);
    Ok(states)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires empirical simulator data"]
    fn test_new() {
        DiodeLaser::new(64).unwrap();
    }

    #[test]
    #[ignore = "requires empirical simulator data"]
    fn test_emit_current_up() -> eyre::Result<()> {
        let state = LaserState {
            current: 134.4,
            voltage: 46.4,
            frequency: 1.0,
            stable: true,
        };
        let mut laser = DiodeLaser::new(64)?;
        laser.with_laser_state(state);

        let (new_state, _) = laser.emit(Signal::CurrentUp);
        println!("{:?}", new_state);
        Ok(())
    }
}
