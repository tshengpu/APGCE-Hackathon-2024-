# APGCE-Hackathon-2024 - Seismic Fault Image Segmentation

![Background](/img/final_output.png)


# Background 
Faults are cracks within the Earth's crust that may help provide conduits for hydrocarbon migration into a trap or stop hydrocarbon on its migration path. Understanding these faults are crucial for geologists in the oil and gas exploration. The challenge is to build and train a computer vision model that can <b> predict the propagation of faults </b> across a given 2D seismic dataset.

At the end of the 48 hours, we are given 10 minutes to present our solution to the judges.

# Our Approach

![Method](/img/Methodology.JPG)

| Challenges | Steps  |
| ---------- | -----  |
| Knowledge Gap | 1. Upskilling on Seismic Domain Knowledge <br> 2. Upskilling on Image Processing <br> 3. Upskilling on Deep Learning modeling |
| Imbalanced Dataset |1. Image Cropping <br> 2. Enhanced Focal Loss with alpha (focus on minority data) <br> 3. Weighted F1 Score Evaluation Metrics |
| Constrains on computation power <br> (16 MiB GPU) | 1. Code Optimization with coding best practices <br> 2. Image Size Reduction  <br> 3. Image Cropping  <br> 4. Batch Size Hyperparameter tunning|
| Handling Overfitting | 1. Hyperparameter tunning on Learning Rate | 
| Lost of direction and motivation after long working hours | 1. Peer motivation <br> 2. Pair mentoring <br> 3. Pair programming |

## Image enhancement to highlight contours 
![enhancement](/img/image%20enhancement.png)

## Fault Thickening to address sparse target data and improve learning
![thickening](/img/fault%20thickening.png)

## Image Cropping
Models are trained using random 512x512 samples of whole images. When a full scale image is passed into the model wrapper, it is right and bottom padded, and then split into grids of 512x512 dimension. Each grid is predicted independently by the model, and the individual predictions are combined to give a final overall prediction.

# Results
![results](/img/training%20results.png)
Left = Enhanced Focal loss function  <br>
Middle = Weighted F1 Score <br>
Right = Sample prediction on validation dataset  <br>

## Validation on Holdout Data
![holdout](/img/holdout_data.png)
- Looks like more training are required 
- Looks like we have too much penalty on the fault (0.1 for background, 0.9 for fault)

# How to use

### Basic Execution
Open a terminal/cmd, change directory to the repository folder, and run the following command to generate the model file into staging folder.
`cat model/splitted-model/model* > staging/model.pt`

Edit the file path within staging/wrapper.py, and execute in terminal/cmd. If successful, the output image will be produced in the staging folder.

### Data Preparation
A small dataset of 10 images are included in data/raw_XXX folders. Preparation steps and visualization are available in data_prep folder.

### Modeling
Modeling folder contains the code to train and evaluate a model.

# Meet the Team

![team](/img/team5.jpeg) <br>
From left to right: Izzudin Hussein (Data Scientist), Lim Chun Yan (Reservoir Engineer), Max Ooi Wei Xiang (Data Scientist), Teo Sheng Pu (Data Engineer), Zulfadhi Mohd Zaki (Geoscientist)

# Event Information
Date: 16-18 Nov 2024  <br>
Location: Common Ground Bukit Bintang, KL <br>
Orgainized by: 
- Asia Petroleum GeoScience Conference & Exhibition 
- Petronas 

# Resources
1. [Final presentation](/doc/DSGS_101.pdf)
2. [Challenge briefing](/doc/GeoHackathon%202024%20Challenge%20Brief.pdf)
