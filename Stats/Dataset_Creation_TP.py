# -*- coding: utf-8 -*-
"""
Created on Sun May 18 09:23:42 2025

@author: VINICIUSSALES
"""

import matplotlib.cm as cm
import seaborn as sns
import numpy as np 
import os 
from matplotlib.colors import ListedColormap
from tqdm import tqdm 
import rasterio as rio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .Open_Save_Image import Open_Save
from sklearn.model_selection import StratifiedShuffleSplit
from rasterio.errors import NotGeoreferencedWarning
import warnings
# Ignorar o aviso NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

class Dataset_Creation(Open_Save):
    
    def menu(self):
        
        no_value = None
        print('Import Image')
        with rio.open(self.select_file()) as src:
            img = src.read()
        
        if np.ndim(img)> 2:
            
            img = img.squeeze(0)
            

        ext = int(input('Existe Mascara ? \n 1 - yes \n 2 - no \n'))
            
        if ext == 1:
            no_value = int(input('Classe de No Mask \n'))
                        
        return self.position_class(img,no_value)

    
    def position_class(self,label_mask, remove_classe,test_size = 0.7):
        
        
        
        H, W = label_mask.shape
        labels = label_mask.flatten()
        indices = np.arange(H * W)
        img = label_mask.copy().astype(np.float64)
        
        if remove_classe is not None :
            
            mask_valid = labels!= remove_classe
            mask_invalid = label_mask== remove_classe
            labels = labels[mask_valid]
            indices = indices[mask_valid]
            img[mask_invalid] = np.nan
            
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(splitter.split(indices, labels))
        
        pixels_treino = indices[train_idx]
        pixels_teste = indices[test_idx]
        
        output_img_train = np.zeros((H, W), dtype=np.uint8)*np.nan  # tudo branco
        output_img_test = np.zeros((H, W), dtype=np.uint8)*np.nan   # tudo branco

        # Marcar pixels de treino com valor 128 (cinza)
        y_train, x_train = np.unravel_index(pixels_treino, (H, W))
        output_img_train[y_train, x_train] = label_mask[y_train, x_train]
        
        # Marcar pixels de teste com valor 64 (mais escuro)
        y_test, x_test = np.unravel_index(pixels_teste, (H, W))
        output_img_test[y_test, x_test] = label_mask[y_test, x_test]
        
        self.plot(img, output_img_train, output_img_test,test_size)
        
        return output_img_train,output_img_test

    def plot(self,img,output_img_train,output_img_test,test_size):
        
        
        classes_img, counts_img = np.unique(img, return_counts=True)
        classes_train, counts_train = np.unique(output_img_train, return_counts=True)
        classes_test, counts_test = np.unique(output_img_test, return_counts=True)

        # Etapa 3: gerar a visualização
        plt.figure(figsize=(12, 6))
        plt.suptitle(" Distribuição de Pixels por Classe", x=0.5, y=0.95,fontsize=24)
        
        classes, counts = np.unique(classes_train[~np.isnan(classes_train)], return_counts=True)
        cmap = cm.get_cmap('tab20', len(classes))  # ou 'hsv', 'Set1', etc.
        cores = [cmap(int(c)) for c in classes]
        cmap_img = ListedColormap(cores)
        
        # Subplot 1: Imagem de treino
        plt.subplot(2, 3, 1)
        plt.imshow(img, cmap=cmap_img)  # ou você pode colorir com mapa_cor se quiser
        plt.title('Pixels Selecionados: Groud Truth 100%', fontsize=14)
        # plt.legend(handles=legenda, loc='upper right', title='Classes', fontsize=10)
        plt.axis("off")
        
        # Subplot 1: Imagem de treino
        plt.subplot(2, 3, 2)
        plt.imshow(output_img_train, cmap=cmap_img)  # ou você pode colorir com mapa_cor se quiser

        plt.title(f'Pixels Selecionados: Treino {100-test_size *100}%', fontsize=14)
        # plt.legend(handles=legenda, loc='upper right', title='Classes', fontsize=10)
        plt.axis("off")
        
        # Subplot 2: Imagem de teste
        plt.subplot(2, 3, 3)
        plt.imshow(output_img_test, cmap=cmap_img)  # idem acima

        plt.title(f'Pixels Selecionados: Teste {test_size *100}%', fontsize=14)
        # plt.legend(handles=legenda, loc='upper right', title='Classes', fontsize=10)
        plt.axis("off")
        
        plt.subplot(2, 3, 4)
        sns.barplot(x=classes_img, y=(counts_img/counts_img.sum())*100, palette=cores, width=0.6)
        plt.grid()
        plt.xlabel("Classe")
        plt.ylabel("Quantidade de Pixels em %")
        plt.title("Distribuição de Classes no Groud Truth", fontsize=14)
        
        # Subplot 3: Barplot treino
        # Dentro do seu subplot:
        plt.subplot(2, 3, 5)
        sns.barplot(x=classes_train,y=(counts_train/counts_train.sum())*100, palette=cores, width=0.6)
        plt.grid()
        plt.xlabel("Classe")
        plt.ylabel("Quantidade de Pixels %")
        plt.title("Distribuição de Classes no Trainamento", fontsize=14)

        
        # Subplot 4: Barplot teste
        plt.subplot(2, 3, 6)
        sns.barplot(x=classes_test, y=(counts_test/counts_test.sum())*100, palette=cores, width=0.6)
        plt.xlabel("Classe")
        plt.grid()
        plt.ylabel("Quantidade de Pixels %")
        plt.title("Distribuição de Classes no Teste", fontsize=14)
        
        plt.show()
    

    

