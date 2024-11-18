# APGCE-Hackathon-2024
Date: 16-18 Nov 2024  <br>
Location: Common Ground Bukit Bintang, KL <br>
Orgainized by: 
- Asia Petroleum GeoScience Conference & Exhibition 
- Petronas 

![alt text](archive/schedule.png)

## Introduction
### Team name: DSGS101

|Team Member Name| Role in Industry | 
| -------------- | ----- |
| Lim Chun Yan | Reservoir Engineer |
| Zulfadhi Mohd Zaki | Geoscientist |
| Teo Sheng Pu | Data Engineer |
| Max Wei Xiang Ooi | Data Scientist |
| Izzudin Hussein | Data Scientist |

![Alt text](team_photo/WhatsApp%20Image%202024-11-18%20at%2015.21.49.jpeg "Title")

# Challenge Background 
Faults are cracks within the Earth's crust that may help provide conduits for hydrocarbon migration into a trap or stop hydrocarbon on its migration path. Horizons are distinct rock layers that represent different periods of deposition and help geologists track how hydrocarbons move through the subsurface. Understanding these faults are crucial for geologists in the oil and gas exploration.

In 3D seismic, faults are relatively easier to interpret compared to 2D seismic due to data continuity across an area of interest. Similarly, horizons are more confidently tracked in 3D seismic as the continuous data allows geologists to follow the same geological layer across the volume. For examples, in a producing field or exploration in a mature basin. In frontier exploration, geologists oftentimes are constrained with only 2D seismic in the form of discrete lines (inline and crossline).

<b>Challenge 1</b>: build and train a computer vision model that can <b> predict the propagation of faults </b> across a given 2D seismic dataset.

<b>Challenge 2</b>: build and train a computer vision model that can <b> predict the propagation of horizons </b> across a given 2D seismic dataset.

At the end of the 48 hours, you are given 10 minutes to present your solution to the judges. The final product will then be uploaded to and made available in a public repository. 

Full info: 
[Challenge Briefing](/starter_pack/GeoHackathon%202024%20Challenge%20Brief.pdf)

# Our Approach (Challenge 1: Seismic Fault Prediction)

![Alt text](archive/Methodology.JPG "Title")

| Challenges | Steps  |
| ---------- | -----  |
| Knowledge Gap | 1. Upskilling on Seismic Domain Knowledge <br> 2. Upskilling on Image Processing <br> 3. Upskilling on Deep Learning modeling |
| Imbalanced Dataset |1. Image Cropping <br> 2. Enhanced Focal Loss with alpha (focus on minority data) <br> 3. Weighted F1 Score Evaluation Metrics |
| Constrains on computation power <br> (16 MiB GPU) | 1. Code Optimization with coding best practices <br> 2. Image Size Reduction  <br> 3. Image Cropping  <br> 4. Batch Size Hyperparameter tunning|
| Handling Overfitting | 1. Hyperparameter tunning on Learning Rate | 
| Lost of direction and motivation after long working hours | 1. Peer motivation <br> 2. Pair mentoring <br> 3. Pair programming |

## Image enhancement to highlight contours 
![alt text](archive/image%20enhancement.png)

## Fault Thickening to address sparse target data and improve learning
![alt text](archive/fault%20thickening.png)

## Our Results
![alt text](archive/training%20results.png)
Left = Enhanced Focal loss function  <br>
Middle = Weighted F1 Score Accuracy <br>
Right = Sample prediction on validation dataset  <br>

## Validation on holdout data
![alt text](archive/holdout_data.png)
- Looks like more training are required 
- Looks like we have too much penalty on the fault (0.1 for background, 0.9 for fault)


# Sponsors: 
![alt text](archive/WhatsApp%20Image%202024-11-17%20at%2008.25.54.jpeg)
