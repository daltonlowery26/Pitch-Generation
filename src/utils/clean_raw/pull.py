# %% Packages
from pybaseball import statcast

# %% PitchData
df = statcast("2015-01-01", "2026-01-01")
df.to_csv("pitch_2015_2026.csv")
