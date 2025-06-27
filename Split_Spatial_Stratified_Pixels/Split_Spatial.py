# -*- coding: utf-8 -*-
"""
Created on Sun May  4 09:36:35 2025

@author: VINICIUSSALES
"""
import pandas as pd 
from matplotlib import pyplot  as plt 
import seaborn as sns 
import numpy as np 
import os 
import rasterio as rio
from tqdm import tqdm 
from Other.Train_Test_Split import Train_Test_Split
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import seaborn as sns

class SplitSpatialClass(Train_Test_Split):
    
    
    def function(self,ph,pw,test_size = 0.7):
        
        self.no_value = None
        print( 'Imagem das Classes')
        with rio.open(self.select_file()) as inds:
            label=inds.read()
            profile = inds.profile.copy() 

        ext = int(input('Existe Mascara ? \n 1 - yes \n 2 - no \n'))
        if ext == 1:
                self.no_value = int(input('Classe da Mask \n'))
        
        if label.ndim > 2:
            label = label.squeeze(0)
        
        train,test = self.split_image_global(ph, pw, label,test_size)
        train = self.remove_pixels_repetidos_por_patch(train, patch_size=32)
        test = self.remove_pixels_repetidos_por_patch(test, patch_size=32)
        
        self.plot(label.astype(np.float32),train.copy().astype(np.float32),test.copy().astype(np.float32),test_size)
        file = self.selection_dir()
        profile['count']=1
        with rio.open(os.path.join(file,"Train.tif"), "w", **profile) as dst:
            dst.write(train,1)
                    
        with rio.open(os.path.join(file,"Test.tif"), "w", **profile) as dst:
            dst.write(test,1)

    
    def split_image_global(self,ph,pw,label,test_size): # Tamanho do patch global em porcentagem 
        
        matrix_train = np.zeros_like(label)
        matrix_test= np.zeros_like(label)
        h = label.shape[0]
        w = label.shape[1]
        sw = int((label.shape[1]*pw)/100)
        sh = int((label.shape[0]*ph)/100)
        
        for cnt_i, i in tqdm(enumerate(range(0, h, sh))):
         for cnt_j,j in enumerate(range(0, w, sw)):
             
            i_end = min(i + sh, h)
            j_end = min(j + sw, w)
            i_start = max(0, i_end - sh)
            j_start = max(0, j_end - sw)
            tile = label[i_start:i_end, j_start:j_end]
            
            train,test=self.change_stratifiel(tile,test_size)
            matrix_train[i_start:i_end, j_start:j_end] = train
            matrix_test[i_start:i_end, j_start:j_end] = test

        
        return matrix_train,matrix_test
    
    def change_stratifiel(self,tile,test_size):
        
        tile = tile.copy()
        np.random.seed(42)
        H, W = tile.shape
        labels = tile.flatten()
        indices = np.arange(H * W)
        
        classes = np.unique(labels)
        
        if self.no_value is not None :
            
            mask_valid = labels!= self.no_value
            labels = labels[mask_valid]
            indices = indices[mask_valid]
            
            output_img_train = np.zeros((H, W), dtype=np.uint8) 
            output_img_test = np.zeros((H, W), dtype=np.uint8)
             
        else:
            output_img_train = np.zeros((H, W), dtype=np.uint8) +255
            output_img_test = np.zeros((H, W), dtype=np.uint8) +255
        
        if len(classes) == 1:
            
            if self.no_value is not None:
                tile_train = np.zeros_like(tile)
                tile_test = np.zeros_like(tile)
            else:
                tile_train = np.zeros_like(tile)+255
                tile_test = np.zeros_like(tile)+255
            
            indices = np.argwhere(tile == classes[0])
            selecionados = indices[np.random.choice(len(indices), size=int(len(indices)*(1-test_size)), replace=False)]
            
            tile_train[selecionados[:, 0], selecionados[:, 1]] = classes[0]
            tile_test[tile_train!=classes[0]]=classes[0]
            
            return tile_train,tile_test

            
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(splitter.split(indices, labels))
        
        pixels_treino = indices[train_idx]
        pixels_teste = indices[test_idx]
        
        y_train, x_train = np.unravel_index(pixels_treino, (H, W))
        output_img_train[y_train, x_train] = tile[y_train, x_train]
        
        y_test, x_test = np.unravel_index(pixels_teste, (H, W))
        output_img_test[y_test, x_test] = tile[y_test, x_test]
       
        return output_img_train,output_img_test
    
    def remove_pixels_repetidos_por_patch(self,label_img, patch_size=64):

        img_filtrada = np.zeros_like(label_img) + 255  # 255 como valor "no data"
    
        h, w = label_img.shape
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = label_img[i:i+patch_size, j:j+patch_size]
                unique_classes = np.unique(patch[patch != 255])
                for c in unique_classes:
                    coords = np.argwhere(patch == c)
                    if len(coords) > 0:
                        y, x = coords[np.random.randint(len(coords))]  # escolhe um pixel aleatório
                        img_filtrada[i + y, j + x] = c
    
        return img_filtrada
    
    def plot(self,img,output_img_train,output_img_test,test_size):
        
        print(np.unique(img, return_counts=True))
        
        if self.no_value is not None:
            
            mask_invalid = img== self.no_value
            output_img_train[mask_invalid] = np.nan
            output_img_test[mask_invalid]= np.nan
        
        else:
            

            output_img_train[output_img_train==255] = np.nan
            output_img_test[output_img_test==255]= np.nan
        
        print(np.unique(output_img_train, return_counts=True))
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
    



