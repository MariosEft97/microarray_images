#!/usr/bin/env python
# coding: utf-8

def image_import(image_name):
    import os
    from skimage import io
    
    path = os.getcwd()
    filename = os.path.join(path, image_name)
    image = io.imread(filename)
    return image

def image_info(image):
    print(f"Type: {type(image)}")
    print(f"Value type: {image.dtype}")
    print(f"Dimensions: {image.shape}")
    print(f"Minimum value: {image.min()}")
    print(f"Maximum value: {image.max()}")
    print(f"Pixels: {image.size}")

def image_view(image, color):
    import matplotlib.pyplot as plt
    
    if color == "RGB":
        plt.imshow(image)
        plt.colorbar();
    elif color == "Grayscale":
        plt.imshow(image, cmap="gray")
        plt.colorbar();
        
    
def rgb2gray(image, scale):
    from skimage import color, img_as_float, img_as_ubyte
    
    rgb = image
    if scale == "byte":
        gray = img_as_ubyte(color.rgb2gray(rgb))
        return gray
    elif scale == "float":
        gray = img_as_float(color.rgb2gray(rgb))
        return gray
    else:
        print("Scale is wrongly specified (byte or float).")

def histogram(image, color, scale):
    import matplotlib.pyplot as plt
    from skimage import exposure
    
    if color == "RGB" and scale == "byte":
        hist_phase, bins_phase = exposure.histogram(image.flatten())
        plt.fill_between(bins_phase, hist_phase, alpha=0.5)
        plt.title('RGB histogram')
        plt.xlabel('pixel intensity')
        plt.ylabel('pixel count')
    elif color == "Grayscale" and scale == "byte":
        hist_phase, bins_phase = exposure.histogram(image)
        plt.fill_between(bins_phase, hist_phase, alpha=0.5)
        plt.title('Grayscale (0-255) histogram')
        plt.xlabel('pixel intensity')
        plt.ylabel('pixel count')
    elif color == "Grayscale" and scale == "float":
        hist_phase, bins_phase = exposure.histogram(image)
        plt.fill_between(bins_phase, hist_phase, alpha=0.5)
        plt.title('Grayscale (0-1) histogram')
        plt.xlabel('pixel intensity')
        plt.ylabel('pixel count')
    else:
        print("Argumnets are wrongly specified (color = RGB or Grayscale, scale = byte or float).")

        
def image_filter(image, algorithm, parameters, select):
    import numpy as np
    from skimage import filters
    import matplotlib.pyplot as plt
    
    image_dictionary = {}
    
    if algorithm == "Gaussian":
        fig, axs = plt.subplots(nrows=2, ncols=len(parameters)//2, figsize=(10,8))
        axs = axs.ravel()
    
        for ax in axs:
            ax.axis("off")
        
        for i, s in enumerate(parameters):
            gaussian = filters.gaussian(image, s)
            axs[i].imshow(gaussian, cmap="gray")
            axs[i].set_title(str(s))
            image_dictionary.update({s:gaussian})
    
    if algorithm == "median":
        from skimage.morphology import disk
        from skimage import img_as_ubyte
        
        fig, axs = plt.subplots(nrows=2, ncols=len(parameters)//2, figsize=(10,8))
        axs = axs.ravel()
    
        for ax in axs:
            ax.axis("off")
        
        for i, d in enumerate(parameters):
            median = filters.rank.median(img_as_ubyte(image), disk(d))
            axs[i].imshow(median, cmap="gray")
            axs[i].set_title(str(d))
            image_dictionary.update({d:median})
    
    if select == False:
        pass
    elif select == True:
        par = float(input("Specify the desired parameter value: "))
        plt.close()
        plt.imshow(image_dictionary[par], cmap="gray")
        plt.colorbar();
        return image_dictionary[par]


def image_show(image, nrows=1, ncols=1, cmap="gray", **kwargs):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    return fig, ax
    

def circle_points(resolution, center, radius):
    import numpy as np
    
    radians = np.linspace(0,2*np.pi, resolution)
    
    c = center[1] + radius*np.cos(radians)
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c,r]).T

def circular_mask(image, center, radius):
    import numpy as np
    
    height = image.shape[0]
    width = image.shape[1]

    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[1])**2 + (y-center[0])**2)

    mask = dist_from_center <= radius
    return mask
    
def segmentation_active_contour(image, width, height, ipd, resolution, center, radius, max_px_move):
    import skimage.segmentation as seg
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    x_coordinate = center[0]
    y_coordinate = center[1]
    spot_snakes = []
    spot_circles = []
    spot_intensities = []
    
    for h in tqdm(range(height)):
        
        for w in range(width):
            points = circle_points(resolution, center, radius)[:-1]
            snake = seg.active_contour(image, points, max_px_move=max_px_move, coordinates="rc")
            circle_mask = circular_mask(image, center, radius)
            circle_region = np.nonzero(circle_mask)
            circle_intensity = image[circle_region]
            spot_snakes.append(snake)
            spot_circles.append(points)
            spot_intensities.append(circle_intensity)
            center[1] += ipd
        
        center[1] = y_coordinate
        center[0] += ipd
    
    fig, axs = plt.subplots(1,2, figsize=(10,8))
    
    axs[0].axis("off")
    axs[0].set_title("Manually Created Circles")
    axs[0].imshow(image, cmap="gray")
    for circle in spot_circles:
        axs[0].plot(circle[:,0], circle[:,1], "-r", lw=1);
    
    axs[1].axis("off")
    axs[1].set_title("Active Contour Circles")
    axs[1].imshow(image, cmap="gray")
    for snake in spot_snakes:
        axs[1].plot(snake[:,0], snake[:,1], "-b", lw=1);
    
    return spot_circles, spot_snakes, spot_intensities

def segmentation_flood(image, width, height, seed_point, ipd, tolerance):
    import skimage.segmentation as seg
    # import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
        
    x_coordinate = seed_point[0]
    y_coordinate = seed_point[1]
    
    spot_regions = []
    spot_intensity = []
    flood_masks = []
    
    for h in tqdm(range(height)):
        for w in range(width):
            point = tuple(seed_point) 
            flood_mask = seg.flood(image, point, tolerance=tolerance)
            region = np.nonzero(flood_mask)
            intensity=image[region]
            spot_regions.append(region)
            spot_intensity.append(intensity)
            flood_masks.append(flood_mask)
            seed_point[1] += ipd
        
        seed_point[1] = y_coordinate
        seed_point[0] += ipd
    
    # fig, ax = image_show(image)
    # for mask in tqdm(flood_masks):
    #     ax.imshow(mask, alpha=0.3)
        
    return spot_regions, spot_intensity

def image_processing(image, scale, angle, rescale=True, rotation=True):
    import skimage.transform as t
    
    if rescale == True & rotation == False:
        rescaled = t.rescale(image, scale)
        return(rescaled)
    elif rescale == False & rotation == True:
        rotated = t.rotate(image, angle)
        return(rotated)
    elif rescale == True & rotation == True:
        rescaled = t.rescale(image, scale)
        rotated = t.rotate(rescaled, angle)
        return(rotated)
    
    

def image2table(peptide_seq, permutation_seq, spot_intensity, intensity_avg):
    import pandas as pd
    import numpy as np
    
    peptide = list(map(str,peptide_seq))
    permutation = list(map(str, permutation_seq))
    permutation.insert(0, "WT")
    permutation.insert(0, "WT")
    
    if intensity_avg == "mean":
        mean_intensity = []
        for x in spot_intensity:
            mean_intensity.append(np.mean(x))
        
        mean_intensity_table = []
        for i in range(0, len(mean_intensity), 18):
            mean_intensity_table.append(mean_intensity[i:i+18])
        
        df = pd.DataFrame(mean_intensity_table, columns=peptide).set_index(pd.Series(permutation))
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        return df
    
    elif intensity_avg == "median":
        median_intensity = []
        for x in spot_intensity:
            median_intensity.append(np.median(x))
        
        median_intensity_table = []
        for i in range(0, len(median_intensity), 18):
            median_intensity_table.append(median_intensity[i:i+18])
        
        df = pd.DataFrame(median_intensity_table, columns=peptide).set_index(pd.Series(permutation))
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        return df
            
    elif intensity_avg == "mode":
        mode_intensity = []
        for x in spot_intensity:
            mode_intensity.append(np.median(x))
        
        mode_intensity_table = []
        for i in range(0, len(mode_intensity), 18):
            mode_intensity_table.append(mode_intensity[i:i+18])
    
        df = pd.DataFrame(mode_intensity_table, columns=peptide).set_index(pd.Series(permutation))
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        return df
    
    elif intensity_avg == "total":
        total_intensity = []
        for x in spot_intensity:
            total_intensity.append(np.sum(x))
        
        total_intensity_table = []
        for i in range(0, len(total_intensity), 18):
            total_intensity_table.append(total_intensity[i:i+18])
    
        df = pd.DataFrame(total_intensity_table, columns=peptide).set_index(pd.Series(permutation))
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        return df
    

def table_scaler(table, scale, peptide_seq, permutation_seq):
    import random
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import MinMaxScaler
    
    random.seed(12345)
    
    peptide = list(map(str,peptide_seq))
    permutation = list(map(str, permutation_seq))
    permutation.insert(0, "WT")
    permutation.insert(0, "WT")
    
    if scale == "standard":
        array = table.to_numpy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(array)
        df = pd.DataFrame(scaled_data, columns=peptide).set_index(pd.Series(permutation))
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        return df
    
    elif scale == "robust":
        array = table.to_numpy()
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(array)
        df = pd.DataFrame(scaled_data, columns=peptide).set_index(pd.Series(permutation))
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        return df
    
    elif scale == "normalization":
        array = table.to_numpy()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(array)
        df = pd.DataFrame(scaled_data, columns=peptide).set_index(pd.Series(permutation))
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        return df
    
def lir_logo(df, threshold, logo=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from math import isnan
    import logomaker as lm
    
    if logo == True:
        plt.ion()

        df_threshold = df[df<threshold]
        df_threshold_reversed = 1-df_threshold
        df_transpose = df_threshold_reversed.T
        t = df_transpose.reset_index()
        t.drop('index', inplace=True, axis=1)
        t.fillna(0)

        logo = lm.Logo(t.fillna(0), baseline_width=0.1, vpad=0.0, figsize=(8, 8))

        return logo
    
    elif logo == False:
        df_threshold = df[df<threshold]
        df_threshold_reversed = 1-df_threshold
        
        probability_dictionary = {}
        for i in range(len(df_threshold_reversed.columns)):
            rounded = df_threshold_reversed.round(3)
            p_dict = rounded.iloc[:,i].to_dict()
            clean_dict = {k: p_dict[k] for k in p_dict if not isnan(p_dict[k])}
            #print(df_threshold_reversed.columns[i], clean_dict)
            probability_dictionary.update({df_threshold_reversed.columns[i]:clean_dict})
        
        for key, value in probability_dictionary.items():
            print(f"{key}: {value}")
            
        return probability_dictionary
    
        