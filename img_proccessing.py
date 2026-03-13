import numpy as np 
import os
from PIL import Image
import csv

png_name = "1a8211e8.png"
png_name2 = "d99805e6.png"
def png_path(png_name): return os.getcwd() + "/Podatki/ris2026-krog1-ucni/" + png_name

matrix_dim = np.array(Image.open(png_path(png_name))).shape

def separate():
    with open(os.getcwd() + "/Podatki/ucni_set.csv", "r") as f:
        csv_read = csv.reader(f)
        csv_list = list(csv_read)
    sep = {}
    for pic, virus in csv_list:
        if virus not in sep:
            sep[virus] = []
        sep[virus].append(pic)
    return sep

def virusless():
    no_virus = separate()["AO"]
    no_virus_img = np.zeros(shape=matrix_dim)
    
    for v in no_virus:
        img = Image.open(png_path(v))
        img_array = (1/255)*np.array(img) # normalize for stability
        no_virus_img = no_virus_img + img_array
    
    return (1/len(no_virus)) * no_virus_img


def norm(A, B):
    return np.linalg.norm(A - B)

print(norm(virusless(), np.array(Image.open(png_path(png_name2)))))