# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:57:07 2025

@author: VINICIUSSALES
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np 
import os 
import rasterio as rio 
from scipy.io import loadmat
import rasterio
from rasterio.transform import from_origin


class clip_image:
    
    def __init__(self):

        pass
               
    def select_file(self):
                        root = tk.Tk()
                        root.withdraw()
                        return filedialog.askopenfilename(parent=None, 
                                                                  title="Select file", 
                                                                  filetypes=[("Files", "*.mat;")])  
    
    def selection_dir(self):
        
        
        root = tk.Tk()
        root.withdraw()
        return  filedialog.askdirectory(parent=None,title="Select file")
                    
                    
    def input_mat(self):
        
        
        file = self.select_file()

        image = list(loadmat(file).items())[3][1]
        
        if len(image.shape)<3:
            image = np.expand_dims(image, axis=0)
        
        else:

            image = np.transpose(image, (2, 0, 1))  

        transform = from_origin(0, 0, 1, 1)
        
        file = file.split('\\')[-1]
        with rasterio.open(
            os.path.join(self.selection_dir(),file.split('.')[0]+'.tif'),
            'w',
            driver='GTiff',
            height=image.shape[1],
            width=image.shape[2],
            count=image.shape[0],
            dtype=image.dtype,
            transform=transform
        ) as dst:
            dst.write(image)
        


    # https://www.sal.t.u-tokyo.ac.jp/hyperdata/
    
    
a = clip_image()
a.input_mat()