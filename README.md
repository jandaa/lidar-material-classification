# lidar-material-classification
Research project investigating the viability of determining material properties from fullwave-form lidar

## Environment setup

Python 3.8 is required to run the data extraction. To set up the virtual python environment you can use the convenience script using:
```
bash scripts/setup.sh
```

## Preprocessing Data for Training

Training the model requires preprocessing of the data into a format that is easily loaded during training. This step can be run with: `python src/extract_waveforms.py`

The outputs are stored under the folder `preprocessed`

## Training & Evaluation

Training can be performed using the training scripts under the `scripts` folder. The `task` command-line parameter controls the order of tasks to be performed. The options are: train (for training TCN), train_random_forest (for training random forest). Each should display the performance and visualize the outputs.