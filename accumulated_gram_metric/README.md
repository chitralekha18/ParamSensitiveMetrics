# Accumulated Gram Metric Computation (AGM)
# Paper Title: Param-Sensing Metrics: A Study on Parameter Sensitivity of Deep-Feature based Evaluation Metrics for Audio Textures
# Authors: <ANONYMIZED>

## INSTALLATION STEPS
conda create -n venv_agm python=3.8 ipykernel numba
conda activate venv_agm
pip install -r requirements.txt

## RUN CODE
python AGM.py ../samples

This takes the path to a folder as an argument that has a set of wav files of audio textures, that vary over a parameter value.
It considers the first file as anchor and the rest as test, and calculates AGM between anchor and all the test files and plots in a graph.
As the parameter value of the test file moves away from the that of the anchor file, the AGM values also increase.

## RUN NOTEBOOK
AGM_playground.ipynb
Run all the cells and get the same AGM plot as above.