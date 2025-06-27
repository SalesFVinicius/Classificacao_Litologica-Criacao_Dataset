# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:30:35 2025

@author: VINICIUSSALES
"""


import numpy as np 
import os 

import rasterio as rio
from rasterio import windows
import glob
import spectral.io.envi as envi
from rasterio.merge import merge
from tqdm import tqdm
from itertools import product 
from Other.Train_Test_Split import Train_Test_Split
from rasterio.errors import NotGeoreferencedWarning
import warnings
# Ignorar o aviso NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)




class Split_Default(Train_Test_Split):
    
    def split_image_default(self,tile_width, tile_height,margin=0,cont_n=0): # Margin é para gerar dataset a partir da metade do Dataset  e Cont para nomear a partir de um contador 
        
        
            print('Imagem Hiper')
            file_gt = self.select_file()
            print('Imagem Classe')
            file = self.select_file()
            print('Diretorio Onde Será Salvo os Dados')
            
            inp_path = self. creation_dir('Input')
            out_path =self. creation_dir('Output',inp_path.replace('Input',''))
            
            for file,out_path in tqdm(zip([file_gt, file], [inp_path, out_path]), total=2, desc="Processando arquivos"):
                cont = cont_n
                with rio.open(file) as inds:
                    meta = inds.meta.copy()
     
                    if file.endswith('.img'):
                        output_filename = 'img_{}.img'
                    else:
                        output_filename = 'img_{}.tif'
                    
                    for window, transform in self.get_tiles(inds, tile_width, tile_height,margin):
                        # print(window)
                        meta['transform'] = transform
                        meta['width'], meta['height'] = window.width, window.height
                        outpath = os.path.join(out_path,output_filename.format(cont))
                        cont = cont +1
                        
                        if tile_width == window.width and window.height == tile_height:
                            with rio.open(outpath, 'w', **meta) as outds:
                                outds.write(inds.read(window=window))
                                outds.update_tags(x=str(window.col_off), y=str(window.row_off),
                                                  img_x = str(inds.meta['width']),img_y = str(inds.meta['height']),img_z = str(inds.meta['count']))
                                
            
            save = input('Dividir em Dados de Treinamento, Teste e Validação ? \n 1: Sim \n 2: Não \n')
            
            if save == str(1): 
                self.train_validation_test(inp_path,out_path)
                        
                        
    def get_tiles(self, ds, width, height, margin=0):
        
        nols, nrows = ds.meta['width'], ds.meta['height']
        
        # Aplica corte nas bordas
        start_col = margin
        end_col = nols - margin
        start_row = margin
        end_row = nrows - margin
    
        # Garante que não haverá overflow
        offsets = product(
            range(start_col, end_col, width),
            range(start_row, end_row, height)
        )
    
        # Define a nova área útil da imagem
        big_window = windows.Window(
            col_off=margin, row_off=margin,
            width=end_col - start_col,
            height=end_row - start_row
        )

        for col_off, row_off in  offsets:
            window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform
            
    
    def merge_tiles(self, mask = False):
        
            # Cria uma lista de caminhos para os arquivos de tiles
            file = self.selection_dir()
            tile_files = [os.path.join(file,f) for f in os.listdir(file) if f.endswith('.img') or f.endswith('.tif')]
                
            tags = rio.open(tile_files[0]).tags()
            meta = rio.open(tile_files[0]).meta
            

            img = np.zeros((int(tags.get('img_z')),int(tags.get('img_y')),int(tags.get('img_x'))))
            for fp in tqdm(tile_files):
                
                slice_img = rio.open(fp)
                tags = slice_img.tags()
                slice_img_matriz = slice_img.read()
                
                z = int(tags.get('img_z'))
                y = int(tags.get('y'))
                x = int(tags.get('x'))
               
                height = slice_img_matriz.shape[1]
                width = slice_img_matriz.shape[2]
                # print(x,y,z,width,height)    
                
                img[:,y:y+height,x:x+width] = slice_img_matriz
            
            # # Mescla os tiles
            # mosaic, out_trans = merge(src_files_to_mosaic, method = 'first', nodata = -9999)
            
            # Atualiza os metadados com base no primeiro tile (pode-se ajustar se necessário)
            out_meta = slice_img.meta.copy()
            out_meta.update({
                "height": img.shape[1],
                "width": img.shape[2]
            })
            
            
            
            if mask is not False:
                
                print('Open Mask')
                mask= rio.open( self.select_file()).read()
                img[mask==0]=0
            # Escreve o raster mesclado em um novo arquivo
            print('Save Image')
            output_filename = self.save_file()
            with rio.open(output_filename, "w", **out_meta) as dest:
                dest.write(img)