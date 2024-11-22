# Defect Detection Algorithm
There is a growing need of defect detection algorithms/models, specifically in the manufacturing industry for example automobiles, to accurately detect the defects in the manufactured parts. This algorithm compares point clouds of the manufactured part (with possible defects) with the ideal point clouds (with no defect) of the same part. The algorithmhas been tested on several point clouds, the results of which are attached in this file.

## Wavelet Detection
The wavelet algorithm has been used to break the point cloud into different details and extract feature of the point cloud.

## Working
The algorithm makes use of the Iterative Closest Point (ICP) algorithm and ICP roation matrix to align the defected and ideal point clouds for comparison. The anomalies/defects are then extracted by the algorithm and saved as .txt file. The algorithm can also visualize the defects using open3d python libarary for better understanding of the place and kind of defect.

## Results
