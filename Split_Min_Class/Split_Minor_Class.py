# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:31:44 2025

@author: VINICIUSSALES
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np 
import os 
from itertools import product
import rasterio as rio
from rasterio import windows
import glob
import spectral.io.envi as envi
from rasterio.merge import merge
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
import random 
from Other.Train_Test_Split import Train_Test_Split

class SplitMinorClass(Train_Test_Split):
    
    
    def split_image_center(self,tile_width, tile_height,remove_classe = None ,cont = 0,treshold=1): # treshold refere-se sobre a quantidade de classe minima eu quero usar e cont é o id da imagem 
            
            print('Hiper')
            file_gt = self.select_file()
            print('Classe')
            file = self.select_file()
            
            points = self.position_class(file, remove_classe)
            
            
            if file.endswith('.img'):
                output_filename = 'img_{}.img'
            else:
                output_filename = 'img_{}.tif'
                
            if file_gt.endswith('.img'):
                    input_filename = 'img_{}.img'
            else:
                    input_filename = 'img_{}.tif'
                    
                    
            data = int(len(points[0])*treshold)
            
            print('Save Data')
            inp_path = self.creation_dir('Input')
            out_path =self. creation_dir('Output',inp_path.replace('Input',''))
            
            for lista in points:
                
                inp_path = self.creation_dir('Input',inp_path.replace('Input',''))
                out_path =self. creation_dir('Output',inp_path.replace('Input',''))
                amostra = random.sample(lista, data)
      
                cont_p = 0 
                for classe , linha_central, coluna_central in amostra:
                    try :
                        with rio.open(file) as inds:
                            meta = inds.meta.copy()
                            window, transform = self.get_tile_by_center(inds,linha_central, coluna_central, tile_width, tile_height)
    
                            meta['transform'] = transform
                            meta['width'], meta['height'] = window.width, window.height
                            outpath = os.path.join(out_path,output_filename.format(cont))
                            
                            if tile_width == window.width and window.height == tile_height:
                                
                                    proporcao = np.sum(inds.read(window=window) == classe) / (tile_width * tile_height)
                                    if proporcao <0.1 :
                                        continue
                                    else:
                                        
                                        with rio.open(outpath, 'w', **meta) as outds:
                                            outds.write(inds.read(window=window))
                                            outds.update_tags(x=str(window.col_off), y=str(window.row_off),
                                                              img_x = str(inds.meta['width']),img_y = str(inds.meta['height']),img_z = str(inds.meta['count'])  )
                            
                                        with rio.open(file_gt) as inds_gt:
                                            meta_gt = inds_gt.meta.copy()
                                    
                                            window_gt, transform_gt = self.get_tile_by_center(inds_gt,linha_central, coluna_central, tile_width, tile_height)
                                        
                                            meta_gt['transform'] = transform
                                            meta_gt['width'], meta_gt['height'] =  window.width, window.height
                                            inpath = os.path.join(inp_path,input_filename.format(cont))
                                            
                                            if tile_width == window_gt.width and window_gt.height == tile_height:
                                                with rio.open(inpath, 'w', **meta_gt) as outds:
                                                    outds.write(inds_gt.read(window=window_gt))
                                                    outds.update_tags(x=str(window_gt.col_off), y=str(window_gt.row_off),
                                                                      img_x = str(inds_gt.meta['width']),img_y = str(inds_gt.meta['height']),img_z = str(inds_gt.meta['count'])  )
                                            cont_p +=1
                                        
                        cont = cont +1
                    except:
                          continue 
                
                self.train_validation_test(inp_path,out_path,inp_path.replace('Input',''),classe_remove=None)
                print(f' \n Classe {classe} quantidade {cont_p}')
                shutil.rmtree(inp_path)
                shutil.rmtree(out_path)
                
    
    def get_tile_by_center(self,ds, linha_central, coluna_central, tile_width, tile_height):
        
         half_w = tile_width // 2
         half_h = tile_height // 2
        
         row_off = linha_central - half_h
         col_off = coluna_central - half_w
        
         # Corrige bordas para manter a janela dentro dos limites
         row_off = max(0, row_off)
         col_off = max(0, col_off)
        
         # Ajusta se ultrapassar os limites da imagem
         if row_off + tile_height > ds.height:
             row_off = ds.height - tile_height
         if col_off + tile_width > ds.width:
             col_off = ds.width - tile_width
        
         # Se mesmo assim não couber (tile maior que a imagem)
         if row_off < 0 or col_off < 0:
             return False, False
        
         window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height)
         transform = rio.windows.transform(window, ds.transform)
        
         return window, transform
    
    
    def position_class(self,file, remove_classe):
        
        with rio.open(file) as inds:
            
            matrix = inds.read()

        labels = matrix.flatten()
        classes, count = np.unique(labels, return_counts = True)
        mask = list(classes[np.argsort(count)])
        
        if remove_classe is not None :
            for i in remove_classe:
                mask.remove(i)   
            
        n_class = np.asarray(mask)  
        
             
        position =[]
        for k in range(n_class.shape[0]):
            position_2 =[]
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[2]):
                    
                    if matrix[0,i,j] == n_class[k] :
                        position_2.append([n_class[k],i,j])
            position.append(position_2)
            
        return position