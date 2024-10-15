import cv2
import numpy as np
from matplotlib import pyplot as plt

from map.fixed_grid_map import FixedGridMap

class SlamGridMap(FixedGridMap):
    def __init__(self, image_path, obstacles=[]):
        # Read the edge image
        self.image_path = image_path
        edges = self.edge_detection(image_path)
        obstacles = self.extract_obstacles(edges, obstacles)

        # Initialize the parent FixedGridMap with the extracted obstacles
        super().__init__(width=self.width, height=self.height, obstacles=obstacles)

    def edge_detection(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 이미지 대비 조정 (히스토그램 평활화)
        equalized_image = cv2.equalizeHist(image)

        # Gaussian Blur로 노이즈 제거
        blurred_image = cv2.GaussianBlur(equalized_image, (11, 11), 0)

        # Canny Edge Detection
        edges = cv2.Canny(blurred_image, 50, 150)
        # _, edges = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)
        self.map_edges = "./map/fig/map_edges.png"
        cv2.imwrite(self.map_edges, cv2.bitwise_not(edges))

        # 1/20 스케일로 축소
        scale_factor = 1 / 20
        new_width = int(edges.shape[1] * scale_factor)
        new_height = int(edges.shape[0] * scale_factor)
        edges = cv2.resize(edges, (new_width, new_height), interpolation=cv2.INTER_AREA)
        edges = cv2.bitwise_not(edges)
        
        self.map_edges_low = "./map/fig/map_edges_low.png"
        cv2.imwrite(self.map_edges_low, edges)
        
        edges = cv2.bitwise_not(edges)
        _, edges = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)
        edges = cv2.bitwise_not(edges)
        
        self.map_edges_low_thres = "./map/fig/map_edges_low_thres.png"
        cv2.imwrite(self.map_edges_low_thres, edges)

        return edges
    
    def extract_obstacles(self, edges, obstacles=None):
        # Invert the image to get edges in white on black background
        edges_inverted = cv2.bitwise_not(edges)
        
        # Threshold to obtain a binary image
        _, thresh = cv2.threshold(edges_inverted, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(f"Number of contours found: {len(contours)}")

        # Grid size matching the edge image
        self.width = edges.shape[1]
        self.height = edges.shape[0]
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 15 and h > 15:
                continue
            obstacles.append({'type': 'rectangle', 'coordinates': [x, y, x + w, y + h]})
        
        return obstacles

    # Modify the plot_map method to include background image
    def plot_map(self, title, path=None):
        # Read and display background image
        bg_image = plt.imread(self.map_edges_low_thres)

        plt.imshow(bg_image, extent=[0, self.width, 0, self.height], origin='lower', cmap=plt.cm.gray)
        
        # # Plot obstacles
        # obstacle_x = [x for x, y in self.obstacles]
        # obstacle_y = [y for x, y in self.obstacles]
        # plt.plot(obstacle_x, obstacle_y, ".k")  # Obstacles as black dots
    
        # Plot rectangle obstacle lines
        outer_lines = [
            [(0, 0), (0, self.height)],
            [(0, 0), (self.width, 0)],
            [(self.width, 0), (self.width, self.height)],
            [(0, self.height), (self.width, self.height)],
        ]
        for line in outer_lines:
            x_values = [line[0][0], line[1][0]]
            y_values = [line[0][1], line[1][1]]
            plt.plot(x_values, y_values, "k-", linewidth=3.0)  # Obstacle lines as black lines
    
        # for line in self.obstacle_lines:
        #     x_values = [line[0][0], line[1][0]]
        #     y_values = [line[0][1], line[1][1]]
        #     plt.plot(x_values, y_values, "k-")

        # # Plot circular obstacles
        # for center_x, center_y, radius in self.circular_obstacles:
        #     circle = plt.Circle((center_x, center_y), radius, color='black', fill=False)
        #     plt.gca().add_patch(circle)
    
        # Plot the path if provided
        if path:
            path_x = [x for x, y in path]
            path_y = [y for x, y in path]
            plt.plot(path_x, path_y, "-or")  # Path as red connected circles

        plt.xlim(-1, self.width + 1)
        plt.ylim(-1, self.height + 1)
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("X [m]", fontsize=16)
        plt.ylabel("Y [m]", fontsize=16)
        # plt.legend(fontsize=15)  

        plt.grid(True)
        plt.axis("equal")    
    
    # Modify the plot_map method to include background image
    def plot_slam_map(self, title, image_path=None, path=None):
        # Read and display background image
        bg_image = plt.imread(image_path)

        plt.imshow(bg_image, extent=[0, self.width, 0, self.height], origin='lower', cmap=plt.cm.gray)
        
        # # Plot obstacles
        # obstacle_x = [x for x, y in self.obstacles]
        # obstacle_y = [y for x, y in self.obstacles]
        # plt.plot(obstacle_x, obstacle_y, ".k")  # Obstacles as black dots
    
        # Plot rectangle obstacle lines
        outer_lines = [
            [(0, 0), (0, self.height)],
            [(0, 0), (self.width, 0)],
            [(self.width, 0), (self.width, self.height)],
            [(0, self.height), (self.width, self.height)],
        ]
        for line in outer_lines:
            x_values = [line[0][0], line[1][0]]
            y_values = [line[0][1], line[1][1]]
            plt.plot(x_values, y_values, "k-", linewidth=3.0)  # Obstacle lines as black lines
    
        # for line in self.obstacle_lines:
        #     x_values = [line[0][0], line[1][0]]
        #     y_values = [line[0][1], line[1][1]]
        #     plt.plot(x_values, y_values, "k-")

        # # Plot circular obstacles
        # for center_x, center_y, radius in self.circular_obstacles:
        #     circle = plt.Circle((center_x, center_y), radius, color='black', fill=False)
        #     plt.gca().add_patch(circle)
    
        # Plot the path if provided
        if path:
            path_x = [x for x, y in path]
            path_y = [y for x, y in path]
            plt.plot(path_x, path_y, "-or")  # Path as red connected circles
    
        plt.xlim(-1, self.width + 1)
        plt.ylim(-1, self.height + 1)
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("X [m]", fontsize=16)
        plt.ylabel("Y [m]", fontsize=16)
        # plt.legend(fontsize=15)  

        plt.grid(True)
        plt.axis("equal")

if __name__ == "__main__":
    # Paths to the images
    map_image_path = "./map/fig/map_slam.png"  # Optional, if you want to include the background image

    # Create an instance of SlamGridMap
    map_instance = SlamGridMap(image_path=map_image_path)

    map_instance.plot_map(title="Grid Map: map_hard")
    plt.savefig("results/grid_map/map_slamgrid.png")

    # Plot the map
    map_instance.plot_slam_map(title="Grid Map: map_hard", image_path=map_image_path)
    plt.savefig("results/grid_map/map_slamgrid.png")

    map_instance.plot_slam_map(title="Grid Map: map_hard", image_path=map_instance.map_edges_low_thres)
    plt.savefig("results/grid_map/map_fixedgrid_map_hard.png")
