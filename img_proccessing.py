import numpy as np 
import os
from PIL import Image
import csv
import cv2
import matplotlib.pyplot as plt

bacterias = ["AO", "Ecoli", "Efae", "Kaer", "Kpne", "Paer", "Saur", "Sepi", "Spyo"]
png_name = "1a8211e8.png"
png_name2 = "1ee5693e.png"
def png_path(png_name): return os.getcwd() + "/Podatki/ris2026-krog1-ucni/" + png_name

matrix_dim = np.array(Image.open(png_path(png_name))).shape

# Loči sliko in vrsto bakterije iz ucni_set.csv in vrne seznam slik, ki pripadejo določeni bakteriji.
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

# Vrne povprečje vseh slik, ki pripada bakteriji "kind".
def avg_bacteria(kind):
    bacteria = separate()[kind]
    bacteria_img = np.zeros(shape=matrix_dim)
    
    for v in bacteria:
        img = Image.open(png_path(v))
        img_array = (1/255)*np.array(img) # normalize for stability
        bacteria_img = bacteria_img + img_array
    
    return (1/len(bacteria)) * bacteria_img


def norm(A, B):
    return np.linalg.norm(A - B)

def classify(image):
    image_matrix = (1/255)*np.array(Image.open(png_path(image)))
    averages = {kind : norm(avg_bacteria(kind), image_matrix) for kind in bacterias}
    return min(averages, key=averages.get)

#print(classify(png_name2))

def find_border(image, threshold = 30):
    img = cv2.imread(str(image))
    if img is None:
        raise ValueError(f"Could not load image: {image}")
    
    # Namesto 3D matrike je 2D matrika
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Izračunamo gradiente/odvode v x in y smeri s pomočjo operatorja Soboljeva
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normaliziramo
    grad_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold
    _, border_mask = cv2.threshold(grad_norm, threshold, 255, cv2.THRESH_BINARY)
    
    return border_mask

mask = find_border(png_path(png_name2), 10)
plt.imshow(mask, cmap='gray')
plt.title("Border Mask: White=255 (border), Black=0 (background)")
plt.show()

np.set_printoptions(threshold=np.inf)
#print(mask)
