# SpeckCollect
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![GitHub repo size](https://img.shields.io/github/repo-size/AdamDHines/SpeckCollect.svg?style=flat-square)

A simple module that collects events from a SynSense Speck2fDevKit and creates temporal frame representations of the collected events. Users can set the time window to collect events over, essentially the 'framerate'. Simply counts the number of events incurred for each pixel (128x128).

Currently, only considers positive events - functionality to be improved in later iterations to include both positive, negative, only negative, and merged events (v0.2.0).

## Usage
Usage is very simple. First, install all the dependencies. Note, that currently `samna` required python <=3.11 so it's recommended you create a new conda environment to avoid issues with other libraries and dependencies.

```console
# Recommended: create new conda environment
conda create -n speckcollect
conda activate speckcollect

# Install dependencies
pip install speckcollect
```

Download the repository and navigate to the repository directory.

```console
# Clone repository
git clone git@github.com:AdamDHines/SpeckCollect.git

# Set current working directory
cd ~/SpeckCollect
```

To run the SpeckCollect module, simply use the `main.py` script which consists of a few arguments. 

	`--time_int` - sets the time in which to collect events over (default 0.033s or 30fps)
	`--directory` - set the directory to save the data to (default ./speckcollect/data)
	`--exp` - set a name for the experiment (default `exp`)
	
```console
# Run the SpeckCollect over a 1s timebin and set the experiment name to TEST001 
python main.py --time_int 1 --exp TEST001
```

The output will include a `TEST001.npy` file of the raw events collected and a folder `TEST001` which contains .png files of frames for the temporal representation of events.
