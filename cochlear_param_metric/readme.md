# Cochlear Param Metric

This folder is the code for calculate the Cochlear Param Metric.

To calculate the distance between two sound files, we have to calculate the statistics parameters defined by McDermott & Simenceli[^1] first, which will be done by McWalter's MATLAB open-source implementation[^2]. The code is also described in detail by McWalter & McDermott 2018[^3]. Then the code will read the statistics from the MATLAB data files, and measure the distance.

## Prerequisites

The following pre-requisites are important to run the example notebooks in this project. 

* Matlab: Please make sure MATLAB is already installed in your system. And "matlab" can be called from command line (means it should be in the PATH environment variable).
    * Install the Matlab 'Add On' Signal Processing Toolbox
    * Install the Matlab 'Add On' Statistics and Machine Learning Toolbox 

* STSstep Matlab code from Richard McWalter's Github. 
    * To get McWalter's code, please go to [this repo: https://github.com/rmcwalter/STSstep](https://github.com/rmcwalter/STSstep). Clone it to current path. 
    * Note that, this "STSstep" also depends on two other projects as below - 

        1. The first one is [minFunc: https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html). Or you can directly download from [here: https://www.cs.ubc.ca/~schmidtm/Software/minFunc_2012.zip](https://www.cs.ubc.ca/~schmidtm/Software/minFunc_2012.zip) please download the code and place them to the folder "_minFunc_2012" inside the STSstep project. 

        2. Another one is [Large time-frequency analysis toolbox (LTFAT): http://ltfat.github.io](http://ltfat.github.io), or directly download from here](https://github.com/ltfat/ltfat/archive/refs/tags/2.4.0.zip) please download the code and place them to the folder "_ltfat" inside the STSstep project. 
        
        For more details on the STSstep project structure, please go to [STSstep's readme](https://github.com/rmcwalter/STSstep#readme).

## Distance Measurement

Please run the `playground.ipynb` to see the distance measurement. 

## References

[^1]: McDermott, J. H. and Simoncelli, E. P. Sound texture perception via statistics of the auditory periphery: evidence from sound synthesis. Neuron, 71(5):926–940, 2011

[^2]: https://github.com/rmcwalter/STSstep 

[^3]: McWalter, R. and McDermott, J. H. Adaptive and selective time averaging of auditory scenes. Current Biology, 28(9):1405–1418, 2018