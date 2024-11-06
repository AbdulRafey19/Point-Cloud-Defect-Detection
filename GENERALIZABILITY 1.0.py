
      
import numpy as np
import os
import open3d as o3d
import pywt 
import math 
from math import cos,sin,atan
import pandas as pd

class wavelet_detection:
    def __init__(self) :
      
      import numpy as np
      import os
      import open3d as o3d
      import pywt 
      self.rotation_matrix2=[]
      self.wrinkles_index=[]
      self.bridging_index=[]
      self.mapped_diff_x=[]
      self.mapped_diff_y=[]
      self.mapped_diff_z=[]
      self.mapped_std_x=[]
      self.mapped_std_y=[]
      self.mapped_std_z=[]
      self.index_original=[]
      self.scale_factor=[]
      
 
      self.copy=np.array([])

    def estimate_normals(self,pc):
        # Estimate normals (with any search method)
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Orient normals towards a specific viewpoint (e.g., the camera origin [0, 0, 0])
        pc.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

        # # Estimate normals with a specified search radius (adjust radius as needed)
        # pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10))

    def create_3d_text(self,text,position,color=[0,0,0],font_size= 50):
        text_3d = o3d.geometry.TriangleMesh.create_text_3d(
        text,font_size=font_size,depth=1.0,width=1.0,height=1.0,)
        text_3d.paint_uniform_color(color)
        text_3d.translate(position)


    def load_point_cloud_from_txt(self,file_path):
        data = np.loadtxt(file_path, delimiter=';', usecols=(0, 1, 2))  # Assuming XYZ format
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(data)
        return point_cloud

    def pick_points(self,pcd):
        import open3d as o3d
        print("")
        print("1) Please pick at least three correspondences using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    def heatmaps_visualisation(self,data,scaling=False):
        import matplotlib.pyplot as plt
        import open3d as o3d
        import numpy as np
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)

        if scaling:
            scale_matrix = np.identity(4)
            scale_matrix[0, 0] = 10  # Scale along X-axis
            scale_matrix[2, 2] = 10  # Scale along Z-axis
            scale_matrix[1, 1] = 5
            # Apply scaling transformation to the point clouds
            pcd.transform(scale_matrix)

        temp=np.asarray(pcd.points)

        y_values = temp[:, 1]
        y_normalized = 1.5*(y_values - y_values.min()) / (y_values.max() - y_values.min())
        colors = plt.get_cmap('viridis')(y_normalized)

        # Assign colors to the point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        self.estimate_normals(pcd)


        o3d.visualization.draw_geometries([pcd] , point_show_normal=True)


    def draw_registration_result(self,source, target, transformation):
        import copy
        import open3d as o3d
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp], point_show_normal=True)
        return source_temp

    def demo_manual_registration(self,source_pcd,target_pcd):
        import open3d as o3d
        import numpy as np
        
        initial_PCD=np.genfromtxt(source_pcd,usecols=[0,1,2],delimiter=';',dtype=str,skip_header=5)
        final_MESH=np.genfromtxt(target_pcd,usecols=[0,1,2],delimiter=';',dtype=str,skip_header=5)

        source = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector(initial_PCD)

        target = o3d.geometry.PointCloud()

        target.points = o3d.utility.Vector3dVector(final_MESH)
        
        print("Visualization of two point clouds before manual alignment")
        print("\nPRESS + AND - TO ADJUST PIXELS")
        self.draw_registration_result(source, target, np.identity(4))

        # pick points from two point clouds and builds correspondences
        picked_id_source = self.pick_points(source)
        picked_id_target = self.pick_points(target)
        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target

        # estimate rough transformation using correspondences
        print("Compute a rough transform using the correspondences given by user")
        print("\nPRESS + AND - TO ADJUST PIXELS")
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target,o3d.utility.Vector2iVector(corr))
        return source, target,trans_init
   
    def ICP(self,source,target,trans_init):
        
        # Downsample point clouds for faster alignment
        voxel_size = 0.01  # Adjust voxel size as needed
        source_downsampled = source.voxel_down_sample(voxel_size)
        target_downsampled = target.voxel_down_sample(voxel_size)
        import open3d as o3d
        # point-to-point ICP for refinement
        print("Perform point-to-point ICP refinement")
        threshold =10  # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_downsampled, target_downsampled, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
            ,o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000))
        estimated_rotation_matrix = reg_p2p.transformation[:3, :3]
        rotation_matrix = reg_p2p.transformation[:3, :3]
        print("Estimated rotation matrix:")
        print(rotation_matrix)
        self.rotation_matrix2=rotation_matrix
        print("\nPRESS + AND - TO ADJUST PIXELS")
        updated_source=self.draw_registration_result(source, target, reg_p2p.transformation)
        print(reg_p2p)
        print("")
        return updated_source

    def getter(self):
        return self.rotation_matrix2

    def Nearest_Neighbors(self,updated_source,target):
        defected_points=np.asarray(updated_source.points)
        ideal_points = np.asarray(target.points)
        # Use k-D tree for nearest neighbor search
        kd_tree = o3d.geometry.KDTreeFlann(target)

        # Calculate the distance between corresponding points
        distances = np.zeros(len(defected_points))

        for i, point in enumerate(defected_points):
            [_, idx, _] = kd_tree.search_knn_vector_3d(point, 1)
            distances[i] = np.linalg.norm(point - ideal_points[idx[0]])
        return distances,ideal_points

    def defect_visualize(self,Points):
        # Create a point cloud from the data
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.asarray(Points))

        # Visualize the selected points in the point cloud
        o3d.visualization.draw_geometries([point_cloud])

        self.bridging_points=np.asarray(Points)

    def defect_detection(self,source,distances,multiplier):
        
        defected_points=np.asarray(source.points)

        # Calculate mean and standard deviation of distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Define a stricter variable threshold based on mean and standard deviation
        defect_threshold = mean_distance + multiplier * std_distance

        # Filter out defected points
        filtered_points = defected_points[distances <= defect_threshold]
        defected_points_after_filter = defected_points[distances > defect_threshold]

        # Create new point clouds with the filtered points and defected points
        filtered_pc = o3d.geometry.PointCloud()
        filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)

        defected_pc_after_filter = o3d.geometry.PointCloud()
        defected_pc_after_filter.points = o3d.utility.Vector3dVector(defected_points_after_filter)

        return defected_points_after_filter,filtered_points,defected_pc_after_filter,filtered_pc



    def all_visualize(self,defected_pc,ideal_pc,filtered_pc,defected_pc_after_filter):
        # Visualize the point clouds
        defected_pc.paint_uniform_color([0, 0, 1])  # Blue for the original defected point cloud
        ideal_pc.paint_uniform_color([1, 1, 0])  # Green for the ideal point cloud
        filtered_pc.paint_uniform_color([0, 1, 0])  # Blue for the filtered point cloud
        defected_pc_after_filter.paint_uniform_color([1, 0, 0])  # Yellow for the defected points after filter

        o3d.visualization.draw_geometries([defected_pc, ideal_pc, filtered_pc, defected_pc_after_filter])
        o3d.visualization.draw_geometries([defected_pc_after_filter])



    def bridging_detect(self,PCD,MESH):

        '''bridiging_detect
        Args:

        PCD  : The defected point cloud in the '.pcd' format
        MESH : The non defected file in the '.pcd' format

        '''

        import numpy as np
        import open3d as o3d

        
        PCD, MESH, trans_init=self.demo_manual_registration(PCD,MESH)

        updated_PCD=self.ICP(PCD,MESH,trans_init)
         
        distances,ideal_points=self.Nearest_Neighbors(updated_PCD,MESH)
        
        while (True):
            os.system('cls')
            multiplier = input("ENTER TOLERANCE VALUE FOR STRICKER THRESHOLD (-5 TO  5) : ")  # Adjust this value to make the threshold stricter (smaller value means stricter)
            try:
                multiplier = float(multiplier)
                
                if( multiplier <= 5 and multiplier >= -5):
                    defected_points,filtered_points,defected_pc,filter_pc = self.defect_detection(updated_PCD,distances,multiplier) 
                    while(True):
                        os.system('cls')
                        Choice = input("PRESS 1 TO SEE DEFECTED POINTS \
                           \n PRESS 2 TO SEE FILTERED POINTS \
                            \n Press 3 TO VIEW BOTH DEFECTED AND FILTERED POINTS \
                           \n PRESS -1 TO EXIT \
                            \n ENTER YOUR CHOICE : ")
                        if (Choice == '1' ):
                            print("\nPRESS + AND - TO ADJUST PIXELS")
                            self.heatmaps_visualisation(defected_points)
                        elif(Choice == '2'):
                            print("\nPRESS + AND - TO ADJUST PIXELS")
                            self.heatmaps_visualisation(filtered_points)
                        elif(Choice == '3'):
                            print("\nPRESS + AND - TO ADJUST PIXELS")
                            defected_pc.paint_uniform_color([1, 0, 0])  # RED for TRUE DEFECTED
                            filter_pc.paint_uniform_color([0, 1, 0])  # GREEN for TRUE NON DEFECTED
                            o3d.visualization.draw_geometries([defected_pc,filter_pc], point_show_normal=True , window_name="VISUALIZATION")
                            print("\n \n RED FOR DEFECTED \n GREEN FOR NON DEFECTED !!!")
                        elif(Choice == '-1'):
                            print("Exiting")
                            break
                        else:
                            print("\n ERROR! NO DATA TO DISPLAY ! ")
                        
                    CHECK=input("\n DO YOU WANT TO CONTINUE WITH DIFFERENT MULTIPLIER VALUE Y/N ? :")
                    if(CHECK == 'N' or CHECK == "n" or CHECK == "NO" or CHECK == 'no'):
                        # Save the defected points after filter to a txt file with semicolon delimiter
                        write_file = input('ENTER ADDRESS OF FILENAME TO SAVE DEFECTED POINTS (C:\... xyz.txt):')
                
                        np.savetxt(write_file, defected_points, delimiter=';')
                        print("FILE HAS BEEN SAVED !!!")
                        os.system('pause')
                        return defected_pc
                    else:
                        continue
                else:
                    print("\nPLEASE STICK WITH THE BOUNDARY -1 TO 1 ")
                    os.system('pause')
                
            except:
                print("ERROR IN READING THE INPUT !!!")
                os.system('pause')
                continue
        
                
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


    def accuracy(self,predicted_defects_pc,defected_pc_path,threshold):
        
        true_defects_pc_path= input("ENTER TRUE DEFECTED POINT CLOUD COMPLETE ADDRESS TEXT FILE ONLY (C:\... xyz.txt)::  ")
        true_defects_pc = self.load_point_cloud_from_txt(true_defects_pc_path)       
        defected_pc = self.load_point_cloud_from_txt(defected_pc_path)

        true_defects_kdtree = o3d.geometry.KDTreeFlann(true_defects_pc)
        predicted_defects_kdtree = o3d.geometry.KDTreeFlann(predicted_defects_pc)
        defected_kdtree = o3d.geometry.KDTreeFlann(defected_pc)
        tp_points = []
        fp_points = []
        fn_points = []
        tn_points = []
        # True Positives and False Positives
        for p in predicted_defects_pc.points:
            [_, idx, dists] = true_defects_kdtree.search_knn_vector_3d(p, 1)
            if len(dists) > 0 and dists[0] < (int(threshold))**2:
                tp_points.append(p)  # True Positive
            else:
                fp_points.append(p)  # False Positive

        # False Negatives
        for t in true_defects_pc.points:
            [_, idx, dists] = predicted_defects_kdtree.search_knn_vector_3d(t, 1)
            if len(dists) == 0 or dists[0] >= (float(threshold))**2:
                fn_points.append(t)  # False Negative

        # True Negatives
        for d in defected_pc.points:
            [_, idx_pred, dists_pred] = predicted_defects_kdtree.search_knn_vector_3d(d, 1)
            [_, idx_true, dists_true] = true_defects_kdtree.search_knn_vector_3d(d, 1)
            if (len(dists_pred) == 0 or dists_pred[0] >= (float(threshold))**2) and (len(dists_true) == 0 or dists_true[0] >= (float(threshold))**2):
                tn_points.append(d)  # True Negative


        tp = len(tp_points)
        fp = len(fp_points)
        fn = len(fn_points)
        tn = len(tn_points)

        # Confusion matrix
        confusion_matrix = np.array([[tp, fp],
                                     [fn, tn]])

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

        #Print metrics
        print(f"Confusion Matrix:\n{confusion_matrix}")
        print(f"Precision: {precision}")
        # print(f"Recall: {recall}")
        # print(f"F1-Score: {f1_score}")
        # print(f"IoU: {iou}")
        print(f"Accuracy: {accuracy}")

        # Create point clouds for each category
        tp_pc = o3d.geometry.PointCloud()
        tp_pc.points = o3d.utility.Vector3dVector(tp_points)
        tp_pc.paint_uniform_color([1, 0, 0])  # RED for True Positives

        fp_pc = o3d.geometry.PointCloud()
        fp_pc.points = o3d.utility.Vector3dVector(fp_points)
        fp_pc.paint_uniform_color([0, 1, 0])  # GREEN for False Positives

        fn_pc = o3d.geometry.PointCloud()
        fn_pc.points = o3d.utility.Vector3dVector(fn_points)
        fn_pc.paint_uniform_color([0, 0, 1])  # Blue for False Negatives

        tn_pc = o3d.geometry.PointCloud()
        tn_pc.points = o3d.utility.Vector3dVector(tn_points)
        tn_pc.paint_uniform_color([1, 1, 0])  # Yellow for True Negatives

        # Visualize all together
        print("\n RED =  TRUE POSITIVE DEFECTS (CORRECTLY CLASSIFY DEFECT AS DEFECTED) \n GREEN = FALSE POSITIVE DEFECTS (NOT DEFECTED BUT CLASSIFY AS DEFECT) \n BLUE = FALSE NEGATIVE DEFECTS (DEFECTED BUT NOT CLASSIFY AS DEFECT) \n YELLOW = TRUE NEGTIVE (NOT DEFECTED AND NOT CLASSIFY AS DEFECTED) \n"  )
         # Visualize all point clouds with legends
        o3d.visualization.draw_geometries([tp_pc, fp_pc, fn_pc,tn_pc],window_name="Confusion Matrix Visualization")
        self.plot_confusion(tp,fp,fn,tn)

# modify this part 




##################################### main code ############################################## 

if __name__ == '__main__' :

    try:
        file_path = input("ENTER DEFECTED POINT CLOUD COMPLETE ADDRESS TEXT FILE ONLY (C:\... xyz.txt): :")
        simulated_file_path = input("ENTER IDEAL POINT CLOUD COMPLETE ADDRESS TEXT FILE ONLY (C:\... xyz.txt): : ")
        V1 = wavelet_detection()
        Predicted_Defects=V1.bridging_detect(file_path,simulated_file_path)
        os.system('cls')
        CHECK = input("DO YOU WANT TO CHECK ACCURACY Y/N ? : ")
        while(True):
            if (CHECK == 'Y' or CHECK == "y" or CHECK == "YES" or CHECK == 'yes'):
                os.system('cls')
                threshold = input('ENTER TOLERANCE VALUE (-5 , 5) :')
                try:
                    threshold = float(threshold)
                    if (threshold <= 5 and threshold >= -5 ):
                        V1.accuracy(Predicted_Defects,file_path,threshold)
                        CHECK = input("DO YOU WANT TO CHECK ACCURACY WITH DIFFERENT TOLERANCE Y/N ? : ")
                    else:
                        print("STICK IN BOUNDARY OF -5 TO 5 !!!")
                        os.system('pause')
                        continue
                except:
                    print("ERROR IN READING TOLERANCE !!!\n\n")
                    os.system('pause')
                    continue
            else:
                break
        choice=input('was the defected point cloud in same orientation as the ideal point cloud, yes or no')
        if (choice=='no'):
            rotation_matrix=V1.getter()
            print('yes')
            points = np.loadtxt(file_path, dtype = float, delimiter=';',usecols=(0,1,2))
            rotated_point_cloud = np.dot(points, rotation_matrix.T)
            print('yes')
            write_file_1 = input("\n ENTER COMPLETE ADDRESS WITH FILE NAME TO SAVE ICP_ROTATED DEFECTED POINT CLOUD :")
            print('yes')
            np.savetxt(write_file_1, rotated_point_cloud, delimiter=';')
            print("FILE HAS BEEN SAVED !!!")
            #visualise the ICP rotated point cloud
            df = pd.read_csv(write_file_1, delimiter=';', header=None)
            # Save the data in XYZ format
            df.to_csv('rotated.xyz', header=False, index=False, sep=' ')
            # Load the point cloud
            pcd = o3d.io.read_point_cloud(r'rotated.xyz')
            # Visualize the point cloud
            o3d.visualization.draw_geometries([pcd])
            #delete the xyz file
            os.remove('rotated.xyz')              
    except:
        print('ERROR IN READING INPUT !!!\n\n')
        os.system('pause')
        
