# PAF-Prediction
**Project Overview**

This project implements a multi-language collaborative machine learning classification pipeline. By leveraging MATLAB's efficient numerical computation capabilities and Python's machine learning ecosystem, it achieves:
1.	Data preprocessing and feature engineering based on MATLAB.
2.	Stacking ensemble classification models based on Python.

**Component Description**

**1. Data_Preprocessing (MATLAB)**

Function: Data segmentation, noise reduction (UPEMD)


**2. Features_Extract (MATLAB)**

Function: Feature extraction, including one-dimensional and two-dimensional

**1D：**

Main_FeatExt_baseDate_1D：The main function for extracting the characteristics of original data

Function_FeatExt_baseDate_1D：The function for extracting the features of the original data, including inter-period features, amplitude features, frequency features, 
waveform features, SVD features, LBP features, and cross-correlation features.

Struct_to_Vertical：Sort the output into a single column and assign numbers to each item, which will facilitate the batch extraction of features in the future. 

**2D：**

Main_FeatExt_baseDate_2D: Main feature extraction function. Switching the called functions allows toggling between instantaneous features and time-period features.

Function_FeatExt_baseDate_2D_Time: Instantaneous feature extraction function.

Function_FeatExt_baseDate_2D_Wave: Time-period feature extraction function.

Superimposed_Average: Superimposed averaging.

zzt_Current_Curl_Generate_256: Generates isomagnetic maps, current maps, and curl maps data.

ISO_Generate_256: Generates isomagnetic maps.

ISO_Boundary_FeatExt: Isomagnetic map boundary feature extraction function.

ISO_DiPole_FeatExt: Isomagnetic map dipole (two-pole) feature extraction function.

ISO_Gravity_FeatExt: Isomagnetic map center-of-gravity feature extraction function.

ISO_PoleNumber_FeatExt: Isomagnetic map magnetic pole feature extraction function.

ISO_Region1_FeatExt: Isomagnetic map Region 1 feature extraction function.

ISO_Region2_FeatExt: Isomagnetic map Region 2 feature extraction function.

Current_Generate_256: Generates current maps.

Current_Gravity_FeatExt: Current map center-of-gravity feature extraction function.

Current_MCV_FeatExt: Current map MCV (Mean Current Vector) feature extraction function.

Current_Region1_FeatExt: Current map Region 1 feature extraction function.

Current_Region2_FeatExt: Current map Region 2 feature extraction function.

Current_SVD_FeatExt: Current map SVD (Singular Value Decomposition) feature extraction function.

Current_TCV_FeatExt: Current map TCV (Total Current Vector) feature extraction function.

Curl_Generate_256: Generates curl maps.

Curl_Area_FeatExt: Curl map area feature extraction function.

Curl_DiPole_FeatExt: Curl map dipole (two-pole) feature extraction function.

Curl_MCV_FeatExt: Curl map MCV (Mean Curl Vector) feature extraction function.

Struct_to_Vertical：Sort the output into a single column and assign numbers to each item, which will facilitate the batch extraction of features in the future.


**3. Stacking_SGXE (Python)**

Function：Feature Engineering，Model training and prediction，Comparative Study of Multiple Classifiers


**4. Environmental requirements**

MATLAB	R2021a+

Python	3.8+

PythonDependence	numpy, pandas, scikit-learn, xgboost
