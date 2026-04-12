import cv2
import numpy as np
import rasterio
from tqdm import tqdm
import math

# https://www.geeksforgeeks.org/python/python-opencv-distancetransform-function/

file = rasterio.open('data/Annual_NLCD_LndCov_2024_CU_C1V1.tif')
output = rasterio.open('data/distance_to_water.tif', 'w', 
                       width=file.width, height=file.height, 
                       count=1, dtype=file.dtypes[0], driver='GTiff',
                       transform=file.transform, crs=file.crs)

size = file.block_shapes[0][0]
y_max = math.ceil(file.shape[0] / size)
x_max = math.ceil(file.shape[1] / size)
for block, window in tqdm(file.block_windows()):
    # read window
    base = file.read(window=window)
    # collect neighborhood
    data = np.ones((base.shape[0], size*3, size*3), dtype=np.uint8) * 255
    y, x = block
    for i in range(-1, 2):
        for j in range(-1, 2):
            px = x+j
            py = y+i
            # Check top bounds
            if py < 0 or px < 0:
                continue
            # Check bottom bounds
            if py >= y_max or px >= x_max:
                continue
            # Use already loaded base (try and save memory)
            row = (i+1)*size
            col = (j+1)*size
            if i == 0 and j == 0:
                segment = base
            segment = file.read(window=file.block_window(1, py, px))
            # Deal with non-square, smaller windows
            y_bound = segment.shape[1]
            x_bound = segment.shape[2]
            data[0, row:row+y_bound, col:col+x_bound] = segment 
    
    if 11 not in data:
        normalized = np.ones((size*3, size*3), dtype=np.uint8) * 255
    else:
        # transpose data
        data = np.transpose(data)
        # threshold image
        ret, thresh = cv2.threshold(data, 11, 255, cv2.THRESH_BINARY)
        # get distances
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # print image
    #cv2.imshow("Distance",normalized)
    #cv2.imshow("Original",thresh)
    #cv2.waitKey(0)
    # Write to raster
    normalized = normalized[size:size*2, size:size*2]
    normalized = np.expand_dims(normalized, axis=-1)
    normalized = np.transpose(normalized)
    output.write(normalized, window=window)
