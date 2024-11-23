# Defect Detection Algorithm
There is a growing need of defect detection algorithms/models, specifically in the manufacturing industry for example automobiles, to accurately detect the defects in the manufactured parts. This algorithm compares point clouds of the manufactured part (with possible defects) with the ideal point clouds (with no defect) of the same part. The algorithmhas been tested on several point clouds, the results of which are attached in this file.

## Wavelet Detection
The wavelet algorithm has been used to break the point cloud into different details and extract feature of the point cloud.

## Working
The algorithm makes use of the Iterative Closest Point (ICP) algorithm and ICP roation matrix to align the defected and ideal point clouds for comparison. The anomalies/defects are then extracted by the algorithm and saved as .txt file and they are called true defects. The algorithm can also visualize the defects using open3d python libarary for better understanding of the place and kind of defect.

## Results
All the PCD files tested on this algorithm were in .txt format. All files, including the true defect file, output by the algorithm, were visualized using the software CloudCompare for better understanding.

### Ideal Point Cloud of Double Dome
![image](https://github.com/user-attachments/assets/d26dbeee-3180-4117-87a9-8200e7baf148)

### Defected Point Cloud of Double Dome (Case 1)
![image](https://github.com/user-attachments/assets/dc2a8446-5e16-4e93-bc48-c1adcc48edc0)

### True Defect Point Cloud of Double Dome (Output of the defect detection algorithm) (Case 1)
![image](https://github.com/user-attachments/assets/f56bbd36-66d4-45dc-a643-53a33473b4fe)

### Defected Point Cloud of Double Dome (Case 2)
![image](https://github.com/user-attachments/assets/3e5f70f0-51a4-4f35-80c1-4f0bc6e106ae)

### True Defect Point Cloud of Double Dome (Output of the defect detection algorithm) (Case 2)
![image](https://github.com/user-attachments/assets/8df27f0c-943d-4366-9472-ee41a5568f3b)

## Conclusion
This algorithm can be used on any point cloud for defect identification and hence it could be the first step for a major breakthrough specially in the manufacturing industry where this algorithm can be used to identify defected products, improving effciiency of manufacturing. 



