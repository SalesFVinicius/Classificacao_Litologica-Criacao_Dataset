# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:02:34 2025

@author: VINICIUSSALES
"""
import seaborn as sns
import numpy as np 
import os 
from tqdm import tqdm 
import rasterio as rio
import matplotlib.cm as cm
import glob
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from Other.Open_Save_Image import Open_Save
from rasterio.errors import NotGeoreferencedWarning
import warnings
# Ignorar o aviso NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class Stats_Cls(Open_Save):
    
    def count_class_patches(self): 
        print('Import pasta onde se encontram o Dataset')
        caminho = self.selection_dir()
        pastas_output = []
        i = 1
        plt.figure(figsize=(12, 6))
        plt.suptitle(" Distribuição de Patches por Classe", x=0.5, y=0.95,fontsize=24)
        for raiz, subpastas, _ in os.walk(caminho):
            for subpasta in subpastas:
                if 'output' in subpasta.lower():  # ignora maiúsculas/minúsculas
                    pastas_output.append(os.path.join(raiz, subpasta))
        
        ext = int(input('Existe Mascara ? \n 1 - yes \n 2 - no \n'))
        no_value = None    
        if ext == 1:
            no_value = int(input('Classe de No Mask \n'))
        
        for file_HR in pastas_output:
            cont = []
            # try:
            file_join = os.path.join(caminho,file_HR)
            for string in os.listdir(file_join):
                    if string.endswith('.tif') or string.endswith('.img'):
        
                        ind =rio.open(os.path.join(file_join, string)).read().flatten()
                        classes, count = np.unique(ind, return_counts = True)
        
                        cont.append(ind)
                
            classes, count = np.unique(np.asarray(cont), return_counts = True)
            if no_value is not None :
                    
                    mask_valid = classes!= no_value

                    classes = classes[mask_valid]
                    count = count[mask_valid]

            cmap = cm.get_cmap('tab20', len(classes)) 
            count =(count/count.sum())*100
            cores = [cmap(int(c)) for c in classes]
            

            plt.subplot(1, len(pastas_output), i)
            sns.barplot(x=classes, y=count, palette=cores, width=0.6)
            plt.grid()
            plt.xlabel("Classe")
            plt.ylabel("Quantidade de Pixels em %")
            if i == 1:
                plt.title(file_HR.split('\\')[-1], fontsize=14)
            else:
                plt.title(file_HR.split('\\')[-2], fontsize=14)
            i +=1
            
        plt.show()

    
    def count_class_img(self):     
        
        print('Contar Classes')
        file = self.select_file()
        with rio.open(file) as inds:
            
            label_array = inds.read()

        labels = label_array.flatten()
        classes, count = np.unique(labels, return_counts = True)

        count_P =(count/count.sum())*100
        for i,j,k in zip(classes,count_P,count):
            print(f'Classe {i}: Pixel {k} Porcentagem {j:.3f}%')
            
            