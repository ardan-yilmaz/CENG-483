## Introduction to Computer Vision - Take Home Exam 2

## OBJECT RECOGNITION WITH BoF, and SIFT Feature Extractor

### FILES
1. the2.pdf: Contains the homework text, the details of the implemented algorithm can be found here.
2. the2.py: Implements an algorithm to solve the object recognition problem using BoF (Bag of Features), SIFT, and k-NN, whose details are given in the homework text, ie, the2.pfd.
3. the2_report.pdf: Explains the SIFT, Dense-SIFT feature extractors, and Bag of Features algorithm by explaining the construction of vocabulary for images using feature extractors and k-means clustering. Further, it discusses the effects of different parameter choices for SIFT, Dense-SIFT, k-NN, and k-means.
4. the2_data.zip: Contains train, validation, and test data. 

### Usage

Cloning the repository, one can run the2.py script with the option of changing the parameters for SIFT and Dense-SIFT, the number of neighbors for k-NN, and the number of clusters for k-means clustering. The default configuration is dense sift with nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6, step_size = 5, n_clusters = 256, and k=16 (For k nearest neighbors). the2_data.zip contains the train, validation, and test data set, and one can use these directly by unzipping the file. Also, different datasets can be used, whose paths are given in the the2.py script between lines 29-32.

For details, please check the homework text, and the report. (the2.pdf, and the2_report.pdf)


