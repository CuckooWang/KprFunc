# KprFunc
A hybrid-learning AI framework for the prediction of functional propionylation site 

## The description of each source code
### GPS 5.0M.py
The position weight determination (PWD) and scoring matrix optimization (SMO) methods were adopted iteratively to generate the optimal postion weights and similarity matrix
### DNN_final.py
A 4-layer DNN framework was implemented in Keras 2.4.3 (http://github.com/fchollet/keras) to general the final model for the prediciton of propionylation sites based on the parameters determined by GPS 5.0M.py
### MAML.py
A 4-layer DNN framework implemented by a MAML strategy to general the model for the prediciton of functional propionylation sites
### Tools.py
Supported methods for GPS 5.0M.py and DNN_final.py
### demo
A small dataset to demo above codes, including the postive & negative dataset, the BLOSUM62 matrix, the typical weights and models generated by GPS 5.0M.py.

## Software Requirements
### OS Requirements
Above codes have been tested on the following systems:  
Windows: Windows 7, Windos 10  
Linux: CentOS linux 7.8.2003  
### Hardware Requirements
All codes and softwares could run on a "normal" desktop computer, no non-standard hardware is needed

## Installation guide
All codes can run directly on a "normal" computer with Python 3.7.9 installed, no extra installation is required

## Instruction
For users who want to run KprFunc in own computer, you should first get the optimal postion weights and similarity matrix usding GPS 5.0M.py with the positive dataset and negative dataset in /demo, then the best output of GPS 5.0M.py will be adopted for DNN.final to generate the models for the prediction of propionlytion site. Finally, the known functional propionylation sites contained in "functionsite" would be taken as secondary positive data while other propionylation sites as negative data to generate the models for the prediction of functional propionylation sites with MAML.py

## Additional information
Expected run time is depended on the hardwares of your computer. In general, it will take about 1 hour to get the final models.
## Contact
Dr. Yu Xue: xueyu@hust.edu.cn  
Dr. Luoying Zhang: zhangluoying@hust.edu.cn  
Chenwei Wang: wangchenwei@hust.edu.cn  
Ke Shui: shuike@hust.edu.cn  



