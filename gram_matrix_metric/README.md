# Gram-Matrix Metric (GM) and Gram-Matrix Cosine Distance Metric (GMcos)
## INSTALLATION STEPS
conda create -n venv_gm python=3.8 ipykernel numba
conda activate venv_gm
pip install -r requirements.txt

## RUN NOTEBOOK
GM_Calculation.ipynb
The notebook is to calculate GM and GMcos metrics given a set of wav files of audio textures with one varying parameter. It considers the first file as the anchor and the rest as the test. For both methods, they are based on the extracted Gram Matrix of the wav files. GMcos is the average of the cosine distance of the two sets of Gram-matrices, while GM is the average of the MSE Loss of the flattened matrices. As the parameter value of the test file moves away from that of the anchor file, the GMcos and GM metric values increase accordingly.

GM_Playground.ipynb
The notebook is to demonstrate the computation. Run all the cells, the plot will show the metric values.