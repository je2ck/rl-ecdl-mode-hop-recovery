# Simulator Data

This directory should contain the empirical laser frequency-current mapping data files:

- `current_up_stable.json` - Stable states during current increase sweep
- `current_up_unstable.json` - Unstable states during current increase sweep
- `current_down.json` - States during current decrease sweep
- `current_down_minus.json` - States during current decrease (negative offset)
- `current_all.json` - Combined sweep data
- `selected_frequencies.json` - Pre-selected target frequencies

Data files are available from the authors upon request.
Each file contains JSON lines with fields: `current`, `voltage`, `frequency`, `stable`.
