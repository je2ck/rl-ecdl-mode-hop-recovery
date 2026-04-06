pub mod drawer;
pub mod laser;

use crate::drawer::Drawer;
use crate::laser::{DiodeLaser, LaserState, Signal};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass]
struct PyDrawer {
    drawer: Drawer,
}

#[pyclass]
struct PyDiodeLaser {
    laser: DiodeLaser,
}

#[pymethods]
impl PyDrawer {
    #[new]
    pub fn new(current_range: u32) -> Self {
        PyDrawer {
            drawer: Drawer::new(current_range).unwrap(),
        }
    }

    pub fn dump(&self) {
        self.drawer.dump().unwrap()
    }

    pub fn update(&mut self, laser_state: LaserState, signal: Signal, amount: f64) {
        self.drawer.update(laser_state, signal, amount);
    }

    pub fn draw<'py>(&self, py: Python<'py>, laser_state: LaserState) -> Bound<'py, PyBytes> {
        let raw_data = self.drawer.draw(laser_state);
        PyBytes::new(py, &raw_data)
    }

    pub fn reset(&mut self, laser_state: LaserState) {
        self.drawer.reset(laser_state);
    }

    pub fn state_distance(&self) -> f64 {
        self.drawer.distance_from_target_frequency()
    }

    pub fn with_target_frequency(&mut self, target: f64) {
        self.drawer.with_target_frequency(target)
    }

    pub fn with_relative_frame(&mut self, initial_current: f64) {
        self.drawer.with_relative_frame(initial_current)
    }

    pub fn with_fix_monitor(&mut self, fix_monitor: bool) {
        self.drawer.with_fix_monitor(fix_monitor);
    }

    pub fn select_random_target(&self, stage_count: u32) -> LaserState {
        self.drawer.select_random_target(stage_count)
    }

    pub fn get_target_from(&self, target_frequency: f64) -> LaserState {
        self.drawer.get_target_from(target_frequency)
    }

    pub fn distance_to_target_state(&self, laser_state: LaserState) -> f64 {
        self.drawer.distance_to_target_state(laser_state)
    }
}

#[pymethods]
impl PyDiodeLaser {
    #[new]
    pub fn new(current_range: u32) -> Self {
        PyDiodeLaser {
            laser: DiodeLaser::new(current_range).expect("Failed to initialize DiodeLaser"),
        }
    }

    pub fn emit(&mut self, signal: Signal) -> (LaserState, f64) {
        self.laser.emit(signal)
    }

    pub fn latest(&self) -> LaserState {
        self.laser.internal_state.latest_state.clone()
    }

    pub fn rand_state(
        &self,
        seed: u64,
        lower: f64,
        upper: f64,
        target_frequency: f64,
        validate: bool,
    ) -> LaserState {
        if seed == 0 {
            if validate {
                return self
                    .laser
                    .choose_random_state(lower, upper, target_frequency, 20);
            }
            return self
                .laser
                .choose_random_state(lower, upper, target_frequency, 10);
        }
        self.laser
            .choose_random_state_with_seed(seed, lower, upper, target_frequency)
    }

    pub fn reset(&mut self, state: LaserState) {
        self.laser.reset(state);
    }

    pub fn get_laser_state(&self, current: f64) -> LaserState {
        self.laser.choose_state_with_current(current)
    }

    pub fn move_horizontal(&mut self, distance: f64) {
        self.laser.move_horizontal(distance)
    }

    pub fn move_vertical(&mut self, distance: f64) {
        self.laser.move_vertical(distance)
    }
}

#[pymethods]
impl LaserState {
    #[new]
    fn new(frequency: f64, current: f64, voltage: f64, stable: bool) -> Self {
        LaserState {
            frequency,
            current,
            voltage,
            stable,
        }
    }
}

#[pymodule]
fn laser_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDrawer>()?;
    m.add_class::<PyDiodeLaser>()?;
    m.add_class::<LaserState>()?;
    m.add_class::<Signal>()?;
    Ok(())
}
