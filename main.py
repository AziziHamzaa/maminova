import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop, rotate
from torch import Tensor
import numpy as np
import tkinter
import matplotlib
import imageio 
import os
import tkinter as tk
import threading
import time
import warnings
matplotlib.use('TkAgg')

from warping import Warping

#ATEX images

before_number = 1
after_number = 2

#after_image_name = f"data_inconel\{after_number}.png"
#before_image_name = f"data_inconel\{before_number}.png"

# CSV Images
#before_image_name = f"Inconel_data/pyInconel_{before_number}.png"
#after_image_name = f"Inconel_data/pyInconel_{after_number}.png"

before_image_name = "All Euler_before_test.png"
after_image_name = "All Euler_after_test.png"


#before_number = 3
#after_number = 4

#before_image_name = f"Black_pyInconel_{before_number}.png"
#after_image_name = f"Black_pyInconel_{after_number}.png"

#before_image_name = f"data/test_images/before/All Euler_before_patch1.png"
#after_image_name  = f"data/test_images/after/All Euler_after_patch1.png"



after =  cv2.imread(after_image_name)
before = cv2.imread(before_image_name)


Is = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
Ie = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

transform = transforms.ToTensor()

Is = transform(Is).unsqueeze(0)
Ie = transform(Ie).unsqueeze(0)

#start_bbox = (100 + x, 1201 + x, 1201 + x, 100 + x)

#end_bbox =  (100 + x, 1201 + x, 1201 + x, 100 + x)

#(100, 1501, 1501, 100)

#start_bbox = (0, 1449 ,1449, 0)

#end_bbox =  (0, 1449 ,1449, 0)

#start_bbox = (30, 270, 270, 30)

#end_bbox =  (30, 270, 270, 30)

#start_bbox = (30, 270, 270, 30)

#x = 10

#end_bbox =  (30 + x , 270  +x , 270 + x, 30 + x)

#start_bbox = (0, 360, 360, 0)

#end_bbox =  (0, 360, 360, 0)


def log_message(test_name, message):
    with open(f"{test_name}.txt", "a") as log_file:
        log_file.write(message + "\n")


#start_bbox = (20, 299, 299, 20)

"""
x = 500
start_bbox = (99, 1449, 1449, 99)
end_bbox =  (99-x, 1449 +x, 1449+x, 99-x)
"""
"""
inconel_number = 369
laminage_number = 1449
number = laminage_number

x_s, y_s = 0, 0
start_bbox = (0 + x_s, number + y_s, number + x_s, 0+y_s)


x_e, y_e = 110, 0
end_bbox =  (0+ x_e, number + y_e, number +x_e, 0+ y_e)
"""
m , n =  0, 498
x_s, y_s = 0, 0
start_bbox = (m + x_s, n + y_s, n + x_s, m+y_s)


x_e, y_e = 0, 0
end_bbox =  (m + x_e, n + y_e, n +x_e, m + y_e)



#start_bbox = (0, 369, 369, 0)

#end_bbox = (0, 369, 369, 0)

#start_bbox = (0, 300, 300, 0)

#x, y = 0, 0

#end_bbox =  (0, 300, 300, 0)

# warping
    
grid_size = (4, 4)

#lr = 0.001
lr = 5

num_epochs = 600


model = Warping(Is, Ie, grid_size, start_bbox, end_bbox)

optimizer = optim.Adam(model.parameters(), lr=lr)

gif_images = []


#test_name = "test_pyInconel_1_2"

#test_name = f"test_Inconel_{before_number}_{after_number}_{time.strftime('%Y-%m-%d %H:%M:%S')}"

#test_name = f"test_traction_{before_number}_{after_number}__{time.strftime('%Y-%m-%d %H:%M:%S')}"
test_name = "pres"

if not os.path.exists(test_name):
    os.makedirs(test_name)

start_time = time.time()

# Log information at the beginning of the script
log_message(test_name ,f"Before Image: {before_image_name}")

log_message(test_name ,f"After Image: {after_image_name}")

log_message(test_name ,f"Start BoundingBox: {start_bbox}")

log_message(test_name ,f"End BoundingBox: {end_bbox}")

log_message(test_name ,f"Number of Epochs: {num_epochs}")

log_message(test_name ,f"Learning Rate: {lr}")

log_message(test_name ,f"Optimizer: {optimizer.__str__()}")

log_message(test_name ,f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

loss = nn.MSELoss()

warnings.filterwarnings("ignore")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    transformed_image, coords = model()

    current_loss = model.loss(coords, rotate(Is[:,:, start_bbox[3]:start_bbox[1], start_bbox[0]:start_bbox[2]], 90), transformed_image)

    
    current_loss.backward()

    optimizer.step()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {current_loss.item()}")
    
    log_message(test_name ,f"Epoch [{epoch + 1}/{num_epochs}], Loss: {current_loss.item()}")

    transformed_image = rotate(transformed_image, -90)
    
    transformed_image = transformed_image.detach().squeeze(0).permute(1, 2, 0).numpy()

    
    first_image = np.zeros_like(Is.detach().squeeze(0).permute(1, 2, 0).numpy())

    second_image = np.zeros_like(Is.detach().squeeze(0).permute(1, 2, 0).numpy())

    third_image = np.zeros_like(Ie.detach().squeeze(0).permute(1, 2, 0).numpy())

    first_image[start_bbox[3]:start_bbox[1],start_bbox[0]:start_bbox[2], :] = transformed_image
   
    second_image = Is.detach().squeeze(0).permute(1, 2, 0).numpy()
    
    third_image = Ie.detach().squeeze(0).permute(1, 2, 0).numpy()
    

    model.plot_grid(grid_size, model.start_grid, model.end_grid, first_image,\
                     start_image = second_image,end_image = third_image,\
                     scaling_factor_deformation_vector=25,\
                     epoch=epoch,image_folder = test_name)
    
    
    if (epoch + 1) % 50 == 0:
        user_input = input("Do you want to continue training? (yes/no): ")
        
        if user_input.lower() == "no":
            plt.imsave('transformed_image.png', transformed_image)
            model.plot_displacement_vector_field(transformed_coords=coords.detach().numpy(), end_bbox=start_bbox, start_grid=model.start_grid, end_image=third_image,\
                                         epoch=epoch,image_folder=test_name, start_image = second_image, step=50, end_grid=model.end_grid)
   
            
            break
        
end_time = time.time()


training_time = end_time - start_time
minutes = int(training_time // 60)
seconds = int(training_time % 60)

print(f"Training time: {minutes} minutes and {seconds} seconds")
log_message(test_name ,f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log_message(test_name ,f"Training time: {minutes} minutes and {seconds} seconds")

output_gif_path = f"{test_name}.gif"
model.create_gif_from_images(test_name, output_gif_path, duration = 100)

