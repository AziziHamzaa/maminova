import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
import skimage.exposure as exposure
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.filters import try_all_threshold, threshold_mean, threshold_isodata
from scipy import ndimage



img = cv2.imread('/home/hamza/Bureau/LORIA/maminova/images/700C/asym1.3-New.tif', cv2.IMREAD_GRAYSCALE)

print(img.shape)



x = 70  # Starting x-coordinate of the ROI
y = 11  # Starting y-coordinate of the ROI
width = 874  # Width of the ROI
height = 927  # Height of the ROI

# Crop the image using array indexing
img = img[y:y+height, x:x+width]




assert img is not None, "file could not be read, check with os.path.exists()"

img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

### Choos
thresh = th2

kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations = 1)
dilated = cv2.dilate(eroded, kernel, iterations = 1)



mask = dilated == 255

s = [[1,1,1],[1,1,1],[1,1,1]]

labeled_mask , num_labels = ndimage.label(mask, structure = s)

print("Number of Grains Detected {}".format(num_labels))


img2 = label2rgb(labeled_mask, bg_label = 0)


fig, ax = plt.subplots(1, 2, figsize=(20, 20))

original_image = cv2.imread('/home/hamza/Bureau/LORIA/maminova/images/700C/asym1.3-New.tif')
original_image = original_image[y:y+height, x:x+width]
ax[0].imshow(original_image)
ax[1].imshow(img2)




plt.show()


properties = ["Area", "orientation"]


# Getting Each Region Properties
clusters = regionprops(labeled_mask, img)

output_file = open('region_properties.txt', 'w')


pixels_to_um = 0.5


# Write the region properties to the file
for cluster_prop in clusters:
    output_file.write(str(cluster_prop["Label"]))
    for _ , prop in enumerate(properties):
        if prop == "Area":
            to_print = cluster_prop[prop]*pixels_to_um**2 # convert pixels into square microns
        elif prop == "orientation":
            to_print = cluster_prop[prop]*57.2958 # convert degree to radians        
        else:
            to_print = cluster_prop[prop]
        output_file.write("," + str(to_print))
    output_file.write("\n")



# Close the output file
output_file.close()

