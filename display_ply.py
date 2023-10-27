from open3d import *    

def main():
    cloud = io.read_point_cloud("exp/mesh/mesh_1.0.ply") # Read point cloud
    visualization.draw_geometries([cloud])    # Visualize point cloud      

if __name__ == "__main__":
    main()