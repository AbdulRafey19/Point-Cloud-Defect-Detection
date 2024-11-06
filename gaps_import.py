import open3d as o3d
import multiprocessing as mp
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import pywt
import os
import copy
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

class GapDetection:
    def __init__(self):
        # Add GLFW initialization
        o3d.visualization.webrtc_server.enable_webrtc()

    def downsample_point_cloud(self, file_path, downsample_factor):
        # Read the entire point cloud
        data = pd.read_csv(file_path, delimiter=";", skiprows=5, header=None).dropna(subset=[0, 1, 2])
        # Convert to NumPy array and downsample
        downsampled_data = np.array(data, dtype=np.float32)[::downsample_factor, :3]
        return downsampled_data

    def process_chunk(self, chunk):
        # Convert chunk to numpy array and compute some simple statistics on the data
        chunk = np.array(chunk).astype(np.float32)
        return chunk

    def process_base(self, downsampled_points):
        points = []
       
        num_processes = 4
        chunk_size = 1000000
        # Read in data in chunks and add to point cloud object using multiprocessing
       
        pool = mp.Pool(num_processes)
        # Assuming downsampled_points is a NumPy array
       
        df = pd.DataFrame(downsampled_points)
        # Process data in chunks
      
        for chunk in df.groupby(np.arange(len(df)) // chunk_size):
            points.append(pool.apply_async(self.process_chunk, args=(chunk[1],)))  # Pass chunk[1] to process_chunk
       
        pool.close()
        pool.join()
        data = np.concatenate([p.get() for p in points])
        return data

    
        

    def compute_density(self, base_points, radius=1):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(base_points)

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        densities = []
        for point in base_points:
            [_, idx, _] = pcd_tree.search_radius_vector_3d(point, radius)
            num_points = len(idx)
            densities.append(num_points if num_points > 0 else 0)
       
        return densities

    def identify_gaps_point_cloud_dbscan(self, point_cloud, density_threshold, eps=1, min_samples=4):
        # Convert the point cloud data to a NumPy array
        X = np.asarray(point_cloud.points)

        # Use DBSCAN to cluster the points
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # Create a KDTree for efficient nearest neighbor searches
        kdtree = cKDTree(X)

        gap_points = []
        
        # Loop through the points and filter out gap points based on density
        for i, point in enumerate(X):
            # Count the number of points within the eps distance of the current point
            num_neighbors = len(kdtree.query_ball_point(point, eps))
           

            # Check if the point is labeled as noise by DBSCAN and has lower density than the threshold
            if labels[i] == -1 and num_neighbors < density_threshold:
                gap_points.append(point)

        gap_points = np.array(gap_points)

        return gap_points

    def surface_reconstruction_base(self, base,  data):
        # Assign color to the original point cloud
        base.paint_uniform_color([0, 1, 1])

        # Compute normals for the point cloud
        base.normals = o3d.utility.Vector3dVector(np.array([[0., 1., 0.] for i in range(len(data))]))
        

        # Duplicate mesh for the underside
        flipped_base = o3d.geometry.PointCloud(base)
        flipped_base.normals = o3d.utility.Vector3dVector(np.array([[0., -1., 0.] for i in range(len(data))]))

        # Poisson Surface Reconstruction
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            d_mesh, d_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(base, depth=13)

        # Remove unwanted plane
        vertices_to_remove = d_densities < np.quantile(d_densities, 0.01)
        d_mesh.remove_vertices_by_mask(vertices_to_remove)

        return base

    def calculate_and_print_gap_statistics(self, base, gap_points):
        total_points = len(base.points)
        gap_point_count = len(gap_points)
        percentage_gaps = (gap_point_count / total_points) * 100

        print("Number of gap points:", gap_point_count)
        print("Total number of points in the original point cloud:", total_points)
        print("Percentage of gaps:", percentage_gaps, "%")

    def visualize_defects(self, base, points):
        defect_point_cloud = o3d.geometry.PointCloud()
        defect_point_cloud.points = o3d.utility.Vector3dVector(points)
        defect_point_cloud.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([defect_point_cloud], point_show_normal=False)
        #o3d.visualization.draw_geometries([defect_point_cloud, base], point_show_normal=False) 

    def fill_low_density_regions(self, base, densities, density_threshold):
        
        # Identify low-density points based on the threshold
        low_density_indices = np.where(densities < density_threshold)[0]
        print ("Low Density indices: ",len(low_density_indices))
        # Create a point cloud from the low-density region
        low_density_pc = o3d.geometry.PointCloud()
        low_density_pc.points = o3d.utility.Vector3dVector(np.asarray(base.points)[low_density_indices])
        print ("Low Density points: ",len(low_density_pc.points))
        # Compute normals for the low-density point cloud
        low_density_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=4))

        # Perform Poisson Surface Reconstruction for the low-density region
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            filled_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(low_density_pc, depth=13)

        return filled_mesh

    def compare_results(self, base, gap_points, filled_mesh):
        # Convert gap points to a binary array
        gap_points_array = np.zeros(len(base.points))
        gap_points_array[np.round(gap_points).astype(int)] = 1

        # Convert filled mesh vertices to a binary array
        filled_points_array = np.zeros(len(base.points))
        filled_points_array[np.asarray(filled_mesh.vertices).astype(int)] = 1


        # Calculate the intersection between gap points and filled points
        intersection = np.logical_and(gap_points_array, filled_points_array)

        # Calculate the percentage of correctly filled gaps
        total_gap_points = gap_points_array.sum()
        correctly_filled_points = intersection.sum()
        percentage_correctly_filled = (correctly_filled_points / total_gap_points) * 100

        print("Correctly identified Gap Points:", total_gap_points)
        print(" Filled Points:", correctly_filled_points)
        print("Percentage Correctly Filled:", percentage_correctly_filled, "%")

        TP = intersection.sum()
        FN = (gap_points_array - intersection).sum()
        FP = (filled_points_array - intersection).sum()
        TN = len(base.points) - (TP + FN + FP)

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / len(base.points)
        f1_score = 2 * (precision * recall) / (precision + recall)

        print("Recall:", recall)
        print("Precision:", precision)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1_score)

        return TP,FP,FN,TN

    def plot_confusion(self, TP, FP, FN, TN):
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import confusion_matrix

        # Create the confusion matrix
        conf_matrix = np.array([[TP, FN], [FP, TN]])

        # Define class labels (for visualization purposes)
        class_names = ['Positive', 'Negative']

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Add counts in the cells
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center", color="black")

        plt.tight_layout()
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.show()




    def gap_func_calling(self, point_cloud_file, downsample_factor):
           
            downsampled_point_cloud = self.downsample_point_cloud(point_cloud_file, downsample_factor)
            # Extract points from downsampled point cloud
            downsampled_points = np.asarray(downsampled_point_cloud)
            data = self.process_base(downsampled_points)
            # Original point cloud object
            base = o3d.geometry.PointCloud()
            base.points = o3d.utility.Vector3dVector(data) 
            o3d.visualization.draw_geometries([base])
            base = self.surface_reconstruction_base(base,data)
            print("________________COMPUTING THE DENSITY ________________")
            densities = self.compute_density(base.points)
            density_threshold = np.mean(densities)
            print("Density threshold: ", density_threshold)
            print("_______________GAPS IDENTIFYING____________________")
            gap_points = self.identify_gaps_point_cloud_dbscan(base, density_threshold)
            self.calculate_and_print_gap_statistics(base, gap_points)
            self.visualize_defects(base, gap_points)
             # Fill low-density regions
            filled_mesh = self.fill_low_density_regions(base, densities, density_threshold)
             # Compare results between gap identification and filled regions
            TP, FP, FN, TN, = self.compare_results(base, gap_points, filled_mesh)
            self.plot_confusion(TP, FP, FN, TN)


            return base, gap_points

if __name__ == "__main__":
    gap_detection = GapDetection()

  