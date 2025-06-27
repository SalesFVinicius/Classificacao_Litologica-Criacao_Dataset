# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:16:14 2025

@author: VINICIUSSALES
"""

from Split_Default import Split_Default
from Stats import Stats
from Split_Min_Class import Split_Minor_Class
from Other import Dataset_Creation_TP
from Split_Spatial_Stratified_Pixels import Split_Spatial
from Clip_Pacthes_Center_Pixel import Clip_Patcher_Center
Stats = Stats.Stats_Cls()

#____________________________________ Split Baseline____________________________________# 
BaseLine = Split_Default()
BaseLine.split_image_default(64,64)
Stats.count_class_patches()

#____________________________________Split_Stratified_Center_Pixel____________________________________# 
Data = Dataset_Creation_TP.Dataset_Creation()
Data.menu()
BaseLine = Split_Default()
BaseLine.split_image_default(64,64)


#____________________________________Split_Spatial_Stratified_Center_Pixel____________________________________# 

sp_spatial = Split_Spatial.SplitSpatialClass()
sp_spatial.function(15,15,test_size = 0.7)
BaseLine = Split_Default()
BaseLine.split_image_default(64,64)
Stats.count_class_patches()

#____________________________________Split_Minor_Classe_Patches____________________________________# 

Sp_Min_Cls = Split_Minor_Class.SplitMinorClass()
Sp_Min_Cls.split_image_center(64, 64,remove_classe=None,treshold=0.7)
Stats.count_class_patches()

#____________________________________Split_Spatial_Stratified_Center_Patches____________________________________# 
sp_spatial = Split_Spatial.SplitSpatialClass()
sp_spatial.function(25,25,test_size = 0.7)
clip = Clip_Patcher_Center.Clip_Patches_Img()
clip.split_image_center(64,64,remove_classe=[255])