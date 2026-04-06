mod drawer;
mod laser;

fn main() {
    println!("Laser simulator ready.");
}

#[cfg(test)]
mod tests {
    use super::drawer::*;
    use super::laser::*;
    use eyre;
    use image::ImageBuffer;
    use image::Rgb;

    #[test]
    fn integration_test_current_up() -> eyre::Result<()> {
        let mut starting_current = 130.0;
        let mut drawer = Drawer::new(64)?;
        let mut laser = DiodeLaser::new(64)?;
        let initial_state = LaserState {
            frequency: 751.54033591,
            current: 130.1,
            voltage: 46.4,
            stable: true,
        };
        laser.with_laser_state(initial_state);

        let mut state = LaserState::default();
        while starting_current < (130.0 + 12.8) {
            starting_current += 0.1;
            (state, _) = laser.emit(Signal::CurrentUp);
            drawer.update(state.clone(), Signal::CurrentUp, 0.0);
        }
        let raw_data = drawer.draw(state);
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_vec(FREQUENCY_RANGE, FREQUENCY_RANGE, raw_data)
                .expect("Failed to create image buffer");

        img.save("integration_test_image.png")
            .expect("Failed to save image");
        Ok(())
    }

    #[test]
    fn integration_test_current_down() -> eyre::Result<()> {
        let mut drawer = Drawer::new(64)?;
        let mut laser = DiodeLaser::new(64)?;
        let initial_state = LaserState {
            frequency: 751.1417348,
            current: 144.3,
            voltage: 46.8,
            stable: false,
        };
        laser.with_laser_state(initial_state);

        let mut state = LaserState::default();
        let mut starting_current = 144.3;
        while starting_current > 130.0 {
            starting_current -= 0.1;
            (state, _) = laser.emit(Signal::CurrentDown);
            drawer.update(state.clone(), Signal::CurrentDown, 0.0);
        }
        let raw_data = drawer.draw(state);
        let distance = drawer.distance_from_target_frequency();
        println!("Distance from target frequency: {}", distance);
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_vec(FREQUENCY_RANGE, FREQUENCY_RANGE, raw_data)
                .expect("Failed to create image buffer");

        img.save("integration_test_image.png")
            .expect("Failed to save image");
        Ok(())
    }

    #[test]
    fn integration_test_pzt_move() -> eyre::Result<()> {
        let mut drawer = Drawer::new(64)?;
        let mut laser = DiodeLaser::new(64)?;
        let initial_state = LaserState {
            frequency: 751.1417348,
            current: 144.3,
            voltage: 46.8,
            stable: false,
        };
        laser.with_laser_state(initial_state);

        let mut state = LaserState::default();
        let mut starting_current = 144.3;
        while starting_current > 130.0 {
            starting_current -= 0.1;
            (state, _) = laser.emit(Signal::CurrentDown);
            drawer.update(state.clone(), Signal::CurrentDown, 0.0);
        }
        let before = drawer.get_all_frequencies();
        (state, _) = laser.emit(Signal::PZTUp);
        drawer.update(state.clone(), Signal::PZTUp, 0.2);
        let after = drawer.get_all_frequencies();
        assert_eq!(before.len(), after.len());
        assert!(before.iter().zip(after.iter()).all(|(b, a)| a >= b));

        let before = after.clone();
        (state, _) = laser.emit(Signal::PZTDown);
        drawer.update(state.clone(), Signal::PZTDown, -0.2);
        let after = drawer.get_all_frequencies();
        assert_eq!(before.len(), after.len());
        assert!(before.iter().zip(after.iter()).all(|(b, a)| a <= b));
        Ok(())
    }
}
