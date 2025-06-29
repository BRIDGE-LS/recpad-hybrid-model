import cv2
from matplotlib import pyplot as plt
import numpy as np


def generate_histogram(image):

    hist_h = cv2.calcHist([image], [0], None, [180], [0,180])
    hist_s = cv2.calcHist([image], [0], None, [255], [0,255])
    hist_v = cv2.calcHist([image], [0], None, [255], [0,255])
    
    plt.subplot(3,1,1)
    plt.plot(hist_h, color="blue")
    plt.xlabel('Matiz (Hue)')
    plt.ylabel('Frequencia')
    plt.xticks(np.arange(0, 180, step=5))

    plt.subplot(3,1,2)
    plt.plot(hist_s, color="green")
    plt.xlabel('Saturacao')
    plt.ylabel('Frequencia')
    plt.xticks(np.arange(0, 255, step=5))

    plt.subplot(3,1,3)
    plt.plot(hist_v, color="red")
    plt.xlabel('Value')
    plt.ylabel('Frequencia')
    plt.xticks(np.arange(0, 255, step=5))

    plt.show()

if __name__ == "__main__":
    
    print("comecou")

    img_adi = cv2.imread("hmu_gc_data/all_image/ADI/ADI_1.png")
    img_deb = cv2.imread("hmu_gc_data/all_image/DEB/DEB_1.png")
    img_lym = cv2.imread("hmu_gc_data/all_image/LYM/LYM_1.png")
    img_muc = cv2.imread("hmu_gc_data/all_image/MUC/MUC_1.png")
    img_mus = cv2.imread("hmu_gc_data/all_image/MUS/MUS_1.png")
    img_nor = cv2.imread("hmu_gc_data/all_image/NOR/NOR_1.png")
    img_str = cv2.imread("hmu_gc_data/all_image/STR/STR_1.png")
    img_tmr = cv2.imread("hmu_gc_data/all_image/TUM/TUM_1.png")

    hsv_img_adi = cv2.cvtColor(img_adi, cv2.COLOR_BGR2HSV)
    hsv_img_deb = cv2.cvtColor(img_deb, cv2.COLOR_BGR2HSV)
    hsv_img_lym = cv2.cvtColor(img_lym, cv2.COLOR_BGR2HSV)
    hsv_img_muc = cv2.cvtColor(img_muc, cv2.COLOR_BGR2HSV)
    hsv_img_mus = cv2.cvtColor(img_mus, cv2.COLOR_BGR2HSV)
    hsv_img_nor = cv2.cvtColor(img_nor, cv2.COLOR_BGR2HSV)
    hsv_img_str = cv2.cvtColor(img_str, cv2.COLOR_BGR2HSV)
    hsv_img_tmr = cv2.cvtColor(img_tmr, cv2.COLOR_BGR2HSV)

    generate_histogram(hsv_img_adi)
    #generate_histogram(hsv_img_deb)
    #generate_histogram(hsv_img_lym)
    #generate_histogram(hsv_img_muc)
    #generate_histogram(hsv_img_mus)
    #generate_histogram(hsv_img_nor)
    #generate_histogram(hsv_img_str)
    #generate_histogram(hsv_img_tmr)
