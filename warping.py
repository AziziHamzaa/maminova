import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
import tkinter
import matplotlib
import numpy as np
from PIL import Image
import os
import random
import imageio
import warnings

matplotlib.use('TkAgg')
 
import itertools



class Warping(nn.Module):

    def __init__(self, Is: Tensor, 
                 Ie: Tensor, grid_size: tuple[int, int], 
                 start_bbox: tuple[int,int,int,int]=None, 
                 end_bbox:tuple[int,int,int,int]=None) -> None:
        """
        Args:
            Is (Tensor): Start Image tensor , image of the material before deformation (1, channel, height, width).
            Ie (Tensor): End Image tensor, image of the material after deformation (1, channel, height, width).
            grid_size (tuple[int, int]): Size of the grid used for warping (height, width).
            start_bbox (tuple[int, int, int, int]): Bounding box for the start image specifying the region of interest for in the start image to align(x_topleft, y_topleft, x_downright, y_downright).
            end_bbox (tuple[int, int, int, int]): Bounding box for the end image specifying the region of interest for in the end image to align (x_topleft, y_topleft, x_downright, y_downright).

        Attributes:
            start_grid (Tensor): Grid placed on the start image generated based on the start bounding box (grid_height, grid_width, 2).
            end_grid (Tensor, Parameter): Grid for warping based on the end bounding box (grid_height, grid_width, 2).
        """
        super(Warping, self).__init__()
        
        self.Is = Is
        self.Ie = Ie
        self.start_bbox = start_bbox
        self.end_bbox = end_bbox
        self.grid_size = grid_size
        self.start_grid = Warping.generate_grid( grid_size = grid_size, bbox = start_bbox)
        self.end_grid = nn.Parameter(Warping.generate_grid(grid_size = grid_size, bbox = end_bbox), requires_grad=True) 
        
    @staticmethod
    def generate_grid(grid_size: tuple[int, int], bbox: tuple[int,int,int,int]) -> Tensor:
        
        """
        Generates a grid based on the given size and bounding box.

        Args:
            grid_size (tuple[int, int]): Size of the grid (height, width).
            bbox (tuple[int, int, int, int]): Bounding box for the image (x_topleft, y_topleft, x_downright, y_downright).

        Returns:
            Tensor: Generated grid with shape (height, width, 2).
        """
        
        n, m = grid_size
    
        i_range = torch.round(torch.linspace(bbox[0], bbox[2], n))
        j_range = torch.round(torch.linspace(bbox[3], bbox[1], m))

        grid = torch.meshgrid(i_range, j_range, indexing="ij")
        grid_tensor = torch.stack(grid, axis=2)

        return grid_tensor 
 
  
    @staticmethod
    def get_quadrilateral(i: int, j: int, grid: Tensor) -> Tensor:
        """
        Returns a quadrilateral from the grid based on the specified indices.

        Args:
            i (int): Index in the x-direction.
            j (int): Index in the y-direction.
            grid (Tensor): Input grid with shape (height, width, 2).

        Returns:
            Tensor: Quadrilateral with shape (2, 2, 2).
        """
        return grid[i:i+2, j:j+2,:]
    
    @staticmethod
    def get_inv_transform_quadrilateral(start_x_range: Tensor, start_y_range: Tensor, end_quad: Tensor) -> Tensor:        
        """
        Computes the inverse transform for a quadrilateral.

        Args:
            start_x_range (Tensor): Range of x-coordinates of pixels in a quadrilateral in the start image with shape Is.size(2)//n.
            start_y_range (Tensor): Range of y-coordinates of pixels in a quadrilateral in the start image with shape Is.size(3)//m.
            end_quad (Tensor): Corresponding quadrilateral in the end image with shape (2, 2, 2).

        Returns:
            Tensor: Inverse transform for a quadrilateral in the grid with shape (Is.size(2)//n, Is.size(3)//m , 2).
        """
        x_00 = start_x_range[0]
        x_11 = start_x_range[-1]
        y_00 = start_y_range[-1]
        y_11 = start_y_range[0]
        

        u = (start_x_range -x_00)/(x_11 - x_00)
        v = (start_y_range -y_00)/(y_11 - y_00)

        q_0 = end_quad[0, 0].T + u.view(u.size(0),1) * (end_quad[0, 1]-end_quad[0, 0])
        q_1 = end_quad[1, 0].T + u.view(u.size(0),1) * (end_quad[1, 1]-end_quad[1, 0])

        
        qx_01 = (q_1 - q_0)[:, 0]
        qy_01 = (q_1 - q_0)[:, 1]

        qx = q_0[:, 0] + torch.outer(v.view(-1), qx_01)
        qy = q_0[:, 1] + torch.outer(v.view(-1), qy_01)

        q_xy = torch.stack([qx, qy], dim=2)

        return q_xy
    
    @staticmethod
    def plot_grid(grid_size: tuple[int, int], start_grid: torch.Tensor, end_grid: torch.Tensor,
                  deformed_image: np.ndarray, start_image: np.ndarray,end_image: np.ndarray,epoch: int,
                   image_folder: str) -> None:
        """
        Constructs a figure with three images (start, end, deformed) and their corresponding grids at each iteration,
        then saves the figure.

        Args:
            grid_size (tuple[int, int]): Size of the grid (height, width).
            start_grid (torch.Tensor): Grid on the start image.
            end_grid (torch.Tensor): Grid on the end image.
            deformed_image (np.ndarray): The deformed end image.
            start_image (np.ndarray): Start image.
            end_image (np.ndarray): End image.
            epoch (int): Current epoch or iteration number.
            image_folder (str): Path to save the figure.
        Returns:
            None
        """
        grid_detached = start_grid.detach().numpy()
        n, m = grid_size
        image_with_grid = np.copy(deformed_image)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))  
        
        for i in range(n - 1):
            for j in range(m - 1):
                quad_points = Warping.get_quadrilateral(i, j, grid_detached).reshape(-1, 2)
                ax3.plot(quad_points[[0, 1, 3, 2, 0], 0], quad_points[[0, 1, 3, 2, 0], 1],   color='black', linestyle='-', linewidth=1)

        
        ax3.set_title(f"Image d'arrivée déformée - itération {epoch}")
        ax3.imshow(image_with_grid)
        
        for i in range(n - 1):
            for j in range(m - 1):
                quad_points = Warping.get_quadrilateral(i, j, grid_detached).reshape(-1, 2)
                ax1.plot(quad_points[[0, 1, 3, 2, 0], 0], quad_points[[0, 1, 3, 2, 0], 1],  color='black', linestyle='-', linewidth=1)
       
        
            
        ax1.set_title('Image de départ')
        ax1.imshow(start_image)

        end_grid_detached = end_grid.detach().numpy()
        
        
        for i in range(n - 1):
            for j in range(m - 1):
                quad_points = Warping.get_quadrilateral(i, j, end_grid_detached).reshape(-1, 2)
                ax2.plot(quad_points[[0, 1, 3, 2, 0], 0], quad_points[[0, 1, 3, 2, 0], 1],  color='black', linestyle='-', linewidth=1)

           
        ax2.set_title(f"Image d'arrivée - iteration {epoch}")
        ax2.imshow(end_image)

        

        plt.tight_layout()

        plt.savefig(f"{image_folder}/{epoch}.png", format='png')
        plt.close()
    """
    @staticmethod
    def calculate_gradients(i, j, start_grid, end_grid, axis = 0):
        
        start_quad = Warping.get_quadrilateral(i, j, start_grid)
        end_quad = Warping.get_quadrilateral(i, j , end_grid)
        
        xD, yD = start_quad[0, 0] 
        xC, yC = start_quad[1, 0]  
        xB, yB = start_quad[1, 1]  
        xA, yA = start_quad[0, 1]
        
        xH, yH = end_quad[0, 0]  
        xG, yG = end_quad[1, 0]  
        xF, yF = end_quad[1, 1]  
        xE, yE = end_quad[0, 1]

        d_x_d_x = lambda x, y: (xF - xE) / (xB - xA) + ((xG - xF) - (xH - xE)) * ((y - yA) / ((yD - yA) * (xB - xA)))
        d_x_d_y = lambda x, y: (xH - xE) / (yD - yA) + ((xG - xF) - (xH - xE)) * ((x - xA) / ((xB - xA) * (yD - yA)))
        d_y_d_x = lambda x, y: (yF - xE) / (xC - xA) + ((yG - xF) - (yH - xE)) * ((y - yA) / ((yD - yA) * (xC - xA)))
        d_y_d_y = lambda x, y: (yH - xE) / (xD - xA) + ((yG - xF) - (yH - xE)) * ((x - xA) / ((xB - xA) * (xD - xA)))
        ''' 
        min_x = min(xD, xC, xB, xA)
        max_x = max(xD, xC, xB, xA)
        min_y = min(yD, yC, yB, yA)
        max_y = max(yD, yC, yB, yA)
        
        if axis==0:
            points = [[x, y, d_x_d_x(x, y), d_x_d_y(x, y)] for x in range(int(min_x), int(max_x) + 1)
                                                            for y in range(int(min_y), int(max_y) + 1)
                                                            if (xD <= x <= xC and yD <= y <= yA) or (xC <= x <= xD and yC <= y <= yB)]
        else:
            points = [[x, y, d_y_d_x(x, y), d_y_d_y(x, y)] for x in range(int(min_x), int(max_x) + 1)
                                                            for y in range(int(min_y), int(max_y) + 1)
        '''                                                    if (xD <= x <= xC and yD <= y <= yA) or (xC <= x <= xD and yC <= y <= yB)]
        
        x_P1 = xA + (xB - xA) // 3 
        y_P1 = yA + (yD - yA) // 3 
        # Calculate the coordinates of P2
        x_P2 = xB + (xC - xB) // 3 
        y_P2 = yB + (yA - yB) // 3 

        # Calculate the coordinates of P3
        x_P3 = xC + (xD - xC) // 3 
        y_P3 = yC + (yB - yC) // 3 

        # Calculate the coordinates of P4
        x_P4 = xD + (xA - xD) // 3 
        y_P4 = yD + (yC - yD) // 3 


        points = [[x_P1, y_P1, d_x_d_x(x_P1, y_P1),d_x_d_y(x_P1, y_P1)],
                  
                  [x_P3, y_P3, d_x_d_x(x_P3, y_P3),d_x_d_y(x_P3, y_P3)],]
        
        
        return points
        (grid_size: tuple[int, int], start_grid: torch.Tensor, end_grid: torch.Tensor,
                  deformed_image: np.ndarray, start_image: np.ndarray,end_image: np.ndarray,epoch: int,
                   image_folder: str, deformation_vectors: bool = False, 
                  scaling_factor_deformation_vector:int = 15) -> None:
    """
    @staticmethod
    def plot_displacement_vector_field(transformed_coords: torch.Tensor, start_grid: torch.Tensor, end_grid:torch.Tensor, 
                                       end_bbox, epoch: int, start_image: np.ndarray, end_image: np.ndarray, 
                                       image_folder: str, step:int=5, grid_size: tuple = (4, 4)) -> None:
        """
        Plot and save the start and end images with displacement field vectors on the start image.

        Args:
            transformed_coords (Tensor): Transformed coordinates of the end image with shape (batch_size, height, width, 2).
            start_grid (Tensor): Starting grid tensor with shape (batch_size, height, width, 2).
            end_grid (Tensor): Ending grid tensor with shape (batch_size, height, width, 2).
            end_bbox (tuple): Bounding box coordinates of the end image (left, bottom, right, top).
            epoch (int): Epoch or iteration number for saving the plot.
            start_image (numpy.ndarray): Numpy array representing the starting image.
            end_image (numpy.ndarray): Numpy array representing the ending image.
            image_folder (str): Path to the folder to save the displacement vector field plot.
            step (int): Step size for quiver plot, default is 5.
            grid_size (tuple): Size of the grid (rows, columns), default is (4, 4).

        Returns:None
        """


        
        if transformed_coords.ndim == 4:
            transformed_coords = transformed_coords.squeeze(0)

        x_end = torch.linspace(end_bbox[0], end_bbox[2], abs(end_bbox[2] - end_bbox[0]))
        y_end = torch.linspace(end_bbox[3], end_bbox[1], abs(end_bbox[3] - end_bbox[1]))
        X_end, Y_end = torch.meshgrid(x_end, y_end)

        end_coords = torch.stack((X_end, Y_end), axis=2)
        end_coords[:,:, 0] = (end_coords[:,:, 0] - (start_image.shape[0]/2))/ (start_image.shape[0]/2)
        end_coords[:,:, 1] = (end_coords[:,:, 1] - (start_image.shape[1]/2))/ (start_image.shape[1]/2)
      

        displacement_vectors =  end_coords.numpy() - transformed_coords 
        
        end_grid_detached = end_grid.detach().numpy()
        start_grid_detached = start_grid.detach().numpy()

        n, m = grid_size

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  
        
        ax1.set_facecolor('black')
        ax2.set_facecolor('black')
        """
        for i in range(n - 1):
            for j in range(m - 1):
                quad_points = Warping.get_quadrilateral(i, j, start_grid_detached).reshape(-1, 2)
                ax1.plot(quad_points[[0, 1, 3, 2, 0], 0], quad_points[[0, 1, 3, 2, 0], 1],  color='#0F52BA', linestyle='-', linewidth=1)

        """
        
        ax1.imshow(start_image)
        
        ax1.quiver( X_end[::step, ::step], Y_end[::step, ::step],
                    displacement_vectors[::step, ::step, 0], displacement_vectors[::step, ::step, 1],
                    scale_units='xy', angles='xy', color='black', linewidths=0.5)
        
        ax1.set_title(f"Image de départ")

        """
        for i in range(n - 1):
            for j in range(m - 1):
                quad_points = Warping.get_quadrilateral(i, j, start_grid_detached).reshape(-1, 2)
                ax2.plot(quad_points[[0, 1, 3, 2, 0], 0], quad_points[[0, 1, 3, 2, 0], 1],   color='#0F52BA', linestyle='-', linewidth=1)
        """
        
        ax2.set_title(f"Image d'arrivée")
        ax2.imshow(end_image)
        
     
    
        plt.tight_layout()

        plt.savefig(f"{image_folder}/displacement_vectors_{epoch}.png", format='png')
        plt.close()


    @staticmethod
   

    def create_gif_from_images(image_folder, output_gif_path, duration = 100):
        """
        Create a GIF file from images saved by plot_grid() method.

        Args:
            image_folder (str): Path to the folder containing the images.
            output_gif_path (str): Path to save the output GIF file.
            duration (int, optional): Duration (in seconds) of each frame in the GIF.

        Returns:
            None
        """
   
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if not filename.startswith('d')]

        
        image_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

        images = []
        for image_file in image_files:
            image = Image.open(image_file)
            images.append(image)

        imageio.mimsave(output_gif_path, images, duration = duration)

        
    @staticmethod
    
    def loss(coords, image1, image2):
        """
        Computes the weighted loss between two images based on the given coordinates.

        Args:
            coords (Tensor): Transformed coordinates of the end_image with shape (batch_size, height, width, 2).
            image1 (Tensor): First image tensor with shape (batch_size, channels, height, width).
            image2 (Tensor): Second image tensor with shape (batch_size, channels, height, width).

        Returns:
            Tensor: Loss value.
        """

        x = coords.squeeze(0)[:, :, 0]
        y = coords.squeeze(0)[:, :, 1]
       
        mask = 1 - ((x < -1) | (x > 1) | (y < -1) | (y > 1)).int()
        
        normalisation_factor = sum(mask.reshape(-1)) * 3

        squeezed_image1 = image1.permute(0, 2, 3, 1).squeeze(0)

        squeezed_image2 = image2.permute(0, 2, 3, 1).squeeze(0)
        
        loss = torch.sum((squeezed_image1 - squeezed_image2)**2 * mask.unsqueeze(2)) / normalisation_factor
        

        
        return loss

    def forward(self):

        n, m = self.grid_size
        row = torch.tensor([])
        col = torch.tensor([])

        for i in range(n-1):

            row = torch.tensor([])

            for j in range(m-1):
                
                start_quad = self.get_quadrilateral(i, j, self.start_grid)
                end_quad = self.get_quadrilateral(i, j , self.end_grid)
                
                start_x_range = torch.round(torch.linspace(start_quad[0,0,0], start_quad[1,0,0], (self.start_bbox[2] - self.start_bbox[0])//(n-1) ))

                start_y_range = torch.round(torch.linspace(start_quad[0,0,1], start_quad[0,1,1], (self.start_bbox[1] - self.start_bbox[3])//(m-1) ))

                #start_x_range = torch.round(torch.arange(start_quad[0,0,0], start_quad[1,0,0], step = 1))

                #start_y_range = torch.round(torch.arange(start_quad[0,0,1], start_quad[0,1,1], step = 1))

                transformed_coords = self.get_inv_transform_quadrilateral(start_x_range, start_y_range, end_quad)
                
                row = torch.cat([row, transformed_coords], dim=1)

            col = torch.cat([row, col], dim=0)
        
        col[:,:, 0] = (col[:,:, 0] - (self.Ie.shape[3]/2))/ (self.Ie.shape[3]/2)
        col[:,:, 1] = (col[:,:, 1] - (self.Ie.shape[2]/2))/ (self.Ie.shape[2]/2)
      
        coords = col.unsqueeze(0)
        
        transformed_image = nn.functional.grid_sample(self.Ie, coords)

        return transformed_image, coords



