RL Project for the event "PI Day" held in Chivasso (TO) on March 14th, 2025.

# Requirements

First, clone the repository and navigate to the project folder:
```
git clone https://github.com/Malgesw/CarRacingAIvsHuman
cd CarRacingAIvsHuman
```
Before installing the dependencies, it’s recommended to create a new virtual environment with Python 3.10 (or a compatible version). For example, using Conda:
```
conda create -n env_name python=3.10
conda activate env_name
```

## Python dependencies
With the virtual environment activated, install the required packages:
```
pip install -r ./requirements.txt
```

# Troubleshooting
If you encounter dependency errors despite having everything installed, you might need to update your ```libgcc``` version:
```
conda install -c conda-forge libstdcxx-ng
```

# How to play
Inside the project directory, with your virtual environment activated, run:
```
python3 game.py
```
