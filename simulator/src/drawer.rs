use image::{ImageBuffer, Rgb};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::from_reader;
use serde_with::{DisplayFromStr, serde_as};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::laser::{data_dir, LaserState, Signal, read_from_file, round_to_decimal_places};

const DEFAULT_TARGET_FREQUENCY: f64 = 751.52630;
pub const FREQUENCY_RANGE: u32 = 1024;

type Current = OrderedFloat<f64>;
type Frequency = OrderedFloat<f64>;

fn get_target_frequency_set(states: Vec<LaserState>) -> eyre::Result<Vec<LaserState>> {
    let abs_path = data_dir();
    let file_path = abs_path.join("selected_frequencies.json");
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let data: Vec<f64> = from_reader(reader)?;
    let target = states
        .into_iter()
        .filter(|state| {
            data.iter().any(|&frequency| {
                round_to_decimal_places(state.frequency, 4) == round_to_decimal_places(frequency, 4)
            })
        })
        .collect_vec();
    Ok(target)
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drawer {
    target_frequency: f64,
    lastest_stable_frequency: f64,
    starting_current: f64,
    starting_frequency: f64,
    #[serde_as(as = "HashMap<DisplayFromStr, DisplayFromStr>")]
    stable_storage: HashMap<Current, Frequency>,
    #[serde_as(as = "HashMap<DisplayFromStr, DisplayFromStr>")]
    unstable_storage: HashMap<Current, Frequency>,
    target_candidates: Vec<LaserState>,

    update_count: usize,
    current_range: u32,
    fix_monitor: bool,
}

impl Drawer {
    pub fn new(current_range: u32) -> eyre::Result<Self> {
        let abs_path = data_dir();
        let dump_file = abs_path.join("../logs/drawer_data.json");

        if dump_file.exists() {
            let mut file = File::open(&dump_file)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            let drawer: Drawer = serde_json::from_str(&contents)?;
            return Ok(drawer);
        }

        let states = read_from_file(abs_path.join("current_up_stable.json"))?;
        let mut stable_storage = HashMap::new();
        states.iter().for_each(|state| {
            stable_storage.insert(OrderedFloat(state.current), OrderedFloat(state.frequency));
        });

        let states = read_from_file(abs_path.join("current_up_unstable.json"))?;
        states.iter().for_each(|state| {
            stable_storage.insert(OrderedFloat(state.current), OrderedFloat(state.frequency));
        });

        let states = read_from_file(abs_path.join("current_down_minus.json"))?;
        states.iter().for_each(|state| {
            stable_storage.insert(OrderedFloat(state.current), OrderedFloat(state.frequency));
        });

        let mut unstable_states = read_from_file(abs_path.join("current_all.json"))?;
        unstable_states.retain(|state| !state.stable);
        let mut unstable_storage = HashMap::new();
        unstable_states.iter().for_each(|state| {
            unstable_storage.insert(OrderedFloat(state.current), OrderedFloat(state.frequency));
        });
        let target_candidates = get_target_frequency_set(states)?;
        Ok(Drawer {
            target_frequency: DEFAULT_TARGET_FREQUENCY,
            lastest_stable_frequency: 0.0,
            starting_current: 120.0,
            starting_frequency: 751.5000,
            stable_storage,
            unstable_storage,
            target_candidates,
            update_count: 0,
            current_range,
            fix_monitor: false,
        })
    }

    pub fn dump(&self) -> eyre::Result<()> {
        let abs_path = data_dir();
        let logs_dir = abs_path.join("../logs");
        std::fs::create_dir_all(&logs_dir)?;
        let mut file = File::create(logs_dir.join("drawer_data.json"))?;
        let json = serde_json::to_string(self).unwrap();
        file.write_all(json.as_bytes())
            .expect("Failed to write to file");
        Ok(())
    }

    pub fn with_target_frequency(&mut self, target_frequency: f64) {
        self.target_frequency = target_frequency;
    }

    pub fn with_relative_frame(&mut self, initial_current: f64) {
        self.starting_current = initial_current - self.current_range as f64 / 20.;
    }

    pub fn with_fix_monitor(&mut self, fix_monitor: bool) {
        self.fix_monitor = fix_monitor;
    }

    pub fn reset(&mut self, laser_state: LaserState) {
        self.push_frequency(laser_state);
    }

    pub fn select_random_target(&self, stage_count: u32) -> LaserState {
        let len_target = self.target_candidates.len();
        self.target_candidates[stage_count as usize % len_target].clone()
    }

    pub fn get_target_from(&self, target_frequency: f64) -> LaserState {
        let abs_path = data_dir();
        let states = read_from_file(abs_path.join("current_down.json")).unwrap();
        states
            .iter()
            .find(|state| {
                round_to_decimal_places(state.frequency, 4)
                    == round_to_decimal_places(target_frequency, 4)
            })
            .unwrap()
            .clone()
    }

    pub fn get_all_frequencies(&self) -> Vec<f64> {
        let mut all_frequencies = Vec::new();
        for (_, frequency) in self.stable_storage.iter() {
            all_frequencies.push(frequency.clone().into());
        }
        for (_, frequency) in self.unstable_storage.iter() {
            all_frequencies.push(frequency.clone().into());
        }
        all_frequencies
    }

    pub fn push_frequency(&mut self, laser_state: LaserState) {
        if laser_state.stable {
            self.push_frequency_inner(laser_state.clone());
            self.lastest_stable_frequency = laser_state.frequency;
            if !self.fix_monitor
                && round_to_decimal_places(laser_state.frequency, 3)
                    == round_to_decimal_places(self.target_frequency, 3)
            {
                self.starting_current = laser_state.current - self.current_range as f64 / 20.;
            }
        } else if !laser_state.stable && self.is_unstable_close_to_stable(laser_state.clone()) {
            self.push_frequency_inner(laser_state.clone());
        }
        if !laser_state.stable && self.lastest_stable_frequency == 0.0 {
            self.lastest_stable_frequency = self.target_frequency;
        }
    }

    fn push_frequency_inner(&mut self, laser_state: LaserState) {
        let storage = if laser_state.stable {
            &mut self.stable_storage
        } else {
            &mut self.unstable_storage
        };
        let current = round_to_decimal_places(laser_state.current, 1);

        if let Some(frequency) = storage.get_mut(&OrderedFloat(current)) {
            *frequency = OrderedFloat(laser_state.frequency);
        } else {
            storage.insert(OrderedFloat(current), OrderedFloat(laser_state.frequency));
        }
    }

    pub fn update(&mut self, laser_state: LaserState, signal: Signal, amount: f64) {
        if matches!(signal, Signal::PZTUp) || matches!(signal, Signal::PZTDown) {
            for (_, frequency) in self
                .stable_storage
                .iter_mut()
                .chain(self.unstable_storage.iter_mut())
            {
                *frequency = OrderedFloat(f64::from(frequency.clone()) + amount);
            }
        }
        self.push_frequency(laser_state);
        self.update_count += 1;
    }

    fn is_unstable_close_to_stable(&self, laser_state: LaserState) -> bool {
        assert_eq!(laser_state.stable, false);

        for (_, frequency) in self.stable_storage.iter() {
            if round_to_decimal_places(frequency.clone().into(), 3)
                == round_to_decimal_places(laser_state.frequency, 3)
            {
                return true;
            }
        }
        false
    }

    pub fn draw(&self, laser_state: LaserState) -> Vec<u8> {
        let mut img =
            ImageBuffer::from_fn(self.current_range, FREQUENCY_RANGE, |_, _| Rgb([0, 0, 0]));

        let white = Rgb([255, 255, 255]);
        let green = Rgb([0, 255, 0]);
        let blue = Rgb([0, 0, 255]);
        let red = Rgb([255, 0, 0]);

        let mut instructions: Vec<(u32, u32, Rgb<u8>)> = Vec::new();

        // Historic frequencies
        let closest = self.closest_frequency();
        for (current, frequency) in self.stable_storage.iter() {
            let x = ((f64::from(*current) - self.starting_current) * 10.).round() as i32;
            let y_raw = ((f64::from(*frequency) - self.starting_frequency) * 10000.) as i32;
            if !self.check_out_of_bounds(x, y_raw) {
                let y = img.height() - 1 - y_raw as u32;
                let color = if let Some(closest) = closest {
                    if round_to_decimal_places((*frequency).into(), 2)
                        == round_to_decimal_places(closest, 2)
                    {
                        green
                    } else {
                        red
                    }
                } else {
                    red
                };
                instructions.push((x as u32, y, color));
            }
        }

        // Current frequency patch
        let x = ((f64::from(laser_state.current) - self.starting_current) * 10.).round() as u32;
        let y_raw = if laser_state.stable {
            std::cmp::min(
                ((laser_state.frequency - self.starting_frequency) * 10000.) as u32,
                1023u32,
            )
        } else {
            std::cmp::min(
                ((self.lastest_stable_frequency - self.starting_frequency) * 10000.) as u32,
                1023u32,
            )
        };
        let y = img.height() - 1 - y_raw;

        let patch_width = 2;
        let patch_height = 10;
        let half_patch_width = (patch_width / 2) as i32;
        let half_patch_height = (patch_height / 2) as i32;

        for dx in 0..patch_width {
            for dy in 0..patch_height {
                let px = x as i32 + dx as i32 - half_patch_width;
                let py = y as i32 + dy as i32 - half_patch_height;
                if px >= 0 && px < img.width() as i32 && py >= 0 && py < img.height() as i32 {
                    instructions.push((px as u32, py as u32, white));
                }
            }
        }

        apply_instructions_parallel(&mut img, &instructions);

        let scaled = scale_and_enlarge(&img, FREQUENCY_RANGE / self.current_range, 64);

        scaled.into_raw()
    }

    fn closest_frequency(&self) -> Option<f64> {
        closest_value(
            &self
                .stable_storage
                .values()
                .map(|f| f.clone().into())
                .collect::<Vec<f64>>(),
            self.target_frequency,
        )
    }

    pub fn closest_current(&self) -> Option<f64> {
        self.stable_storage
            .iter()
            .min_by(|(_, freq_a), (_, freq_b)| {
                let dist_a = (freq_a.into_inner() - self.target_frequency).abs();
                let dist_b = (freq_b.into_inner() - self.target_frequency).abs();
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .map(|(current, _)| current.into_inner())
    }

    pub fn distance_to_target_state(&self, laser_state: LaserState) -> f64 {
        let closest = self.closest_current();
        if let Some(closest) = closest {
            (closest - laser_state.current).abs()
        } else {
            self.current_range as f64
        }
    }

    pub fn distance_from_target_frequency(&self) -> f64 {
        let close_frequencies = if let Some(closest) = self.closest_frequency() {
            self.stable_storage
                .values()
                .filter(|f| {
                    round_to_decimal_places(f64::from(**f), 3)
                        == round_to_decimal_places(closest, 3)
                })
                .collect::<Vec<&Frequency>>()
        } else {
            vec![]
        };

        if close_frequencies.len() < 2 {
            return 1e10;
        }

        let min: f64 = close_frequencies
            .clone()
            .into_iter()
            .min()
            .unwrap()
            .clone()
            .into();
        let max: f64 = close_frequencies.into_iter().max().unwrap().clone().into();
        ((min + max) / 2. - self.target_frequency).abs()
    }

    fn check_out_of_bounds(&self, x: i32, y: i32) -> bool {
        x >= self.current_range as i32 || y >= FREQUENCY_RANGE as i32 || x < 0 || y < 0
    }
}

fn closest_value(vec: &[f64], a: f64) -> Option<f64> {
    vec.into_iter()
        .min_by(|x, y| (*x - a).abs().partial_cmp(&(*y - a).abs()).unwrap())
        .copied()
}

fn scale_and_enlarge(
    image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    n: u32,
    scale_factor: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = image.dimensions();
    let new_width = width * n;
    let mut new_img = ImageBuffer::new(new_width, height);

    let half_scale = (scale_factor as i32 - 1) / 2;

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            for i in 0..n {
                let new_x = x * n + i;
                if *pixel == Rgb([0, 0, 0]) {
                    new_img.put_pixel(new_x, y, *pixel);
                } else {
                    for j in 0..scale_factor {
                        let new_y = y as i32 + j as i32 - half_scale;
                        let new_y = new_y.clamp(0, height as i32 - 1) as u32;
                        new_img.put_pixel(new_x, new_y, *pixel);
                    }
                }
            }
        }
    }
    new_img
}

fn apply_instructions_parallel(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    instructions: &[(u32, u32, Rgb<u8>)],
) {
    let shared_img = Arc::new(Mutex::new(img));

    instructions.par_iter().for_each(|&(x, y, color)| {
        let mut img_lock = shared_img.lock().unwrap();
        img_lock.put_pixel(x, y, color);
    });
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    use super::*;

    #[test]
    fn test_round_f64() {
        assert_eq!(round_to_decimal_places(1.234567, 2), 1.23);
        assert_eq!(round_to_decimal_places(1.234567, 3), 1.235);
        assert_eq!(round_to_decimal_places(1.234567, 4), 1.2346);
    }

    #[test]
    fn test_push_frequency() {
        let mut drawer = Drawer::new(64).unwrap();
        let laser_state = LaserState {
            current: 135.1f64,
            voltage: 46.8f64,
            frequency: 751.54415455f64,
            stable: true,
        };
        drawer.push_frequency_inner(laser_state);
        assert_eq!(
            *drawer.stable_storage.get(&OrderedFloat(135.1)).unwrap(),
            OrderedFloat(751.54415455)
        );
    }

    fn generate_test_drawer() -> Drawer {
        let mut drawer = Drawer::new(64).unwrap();
        let laser_state = LaserState {
            current: 135.1f64,
            voltage: 46.8f64,
            frequency: 751.54415455f64,
            stable: true,
        };
        drawer.push_frequency_inner(laser_state);
        let laser_state = LaserState {
            current: 135.2,
            voltage: 46.8,
            frequency: 751.54405455,
            stable: true,
        };
        drawer.push_frequency_inner(laser_state);
        let laser_state = LaserState {
            current: 135.3,
            voltage: 46.8,
            frequency: 751.54395455,
            stable: true,
        };
        drawer.push_frequency_inner(laser_state);
        drawer
    }

    #[test]
    fn test_generating_image() {
        let mut drawer = generate_test_drawer();
        let signal = Signal::CurrentUp;
        let laser_state = LaserState {
            current: 135.4,
            voltage: 46.8,
            frequency: 751.54385455,
            stable: true,
        };
        drawer.update(laser_state.clone(), signal, 0.0);
        let raw_data = drawer.draw(laser_state);
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_vec(FREQUENCY_RANGE, FREQUENCY_RANGE, raw_data)
                .expect("Failed to create image buffer");

        img.save("drawer_test_image.png")
            .expect("Failed to save image");
    }
}
