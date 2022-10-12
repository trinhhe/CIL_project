import cv2
from numpy.core.defchararray import index
import config
import random
import numpy as np
import os, shutil
from os import listdir
from os.path import isfile, join
from skimage.util import random_noise
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

def rotate(data):

    rotated_data = []
    for [image , ground_truth] in data:
        img = Image.fromarray(image)
        grdt = Image.fromarray(ground_truth)

        angle = random.randint(0 , 360)
        img = img.rotate(angle)
        grdt = grdt.rotate(angle)

        img = np.array(img)
        grdt = np.array(grdt)
        rotated_data.append([img , grdt])
    
    rotated_data = np.array(rotated_data)
    return rotated_data

def translate(data):

    translated_data = []
    a = 1
    b = 0
    c = 0 #left/right (i.e. 5/-5)
    d = 0
    e = 1
    f = 0 #up/down (i.e. 5/-5)
    
    translated_x = data[0][0].shape[1]/4
    translated_y = data[0][0].shape[0]/4
    print("translating x by range [ -%d , %d ] "%(translated_x , translated_x))
    print("translating y by range [ -%d , %d ] "%(translated_y , translated_y))

    for [image , ground_truth] in data:
        img = Image.fromarray(image)
        grdt = Image.fromarray(ground_truth)

        c = random.randint( - translated_x, translated_x )
        f = random.randint( - translated_y, translated_y )
        img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
        grdt = grdt.transform(grdt.size, Image.AFFINE, (a, b, c, d, e, f))

        img = np.array(img)
        grdt = np.array(grdt)
        translated_data.append([img , grdt])
    
    translated_data = np.array(translated_data)
    return translated_data

def add_gaussian_noise(data):
    
    noised_data = []
    for [image, ground_truth] in data:
        noise_img = random_noise(image , mode ="gaussian", var = 0.05**2)
        noise_img = np.array(noise_img)
        noised_data.append([noise_img , ground_truth])
    
    return noised_data

def blur(data):

    blurred_data = []
    for [image, ground_truth] in data:
        img = Image.fromarray(image)
        img = img.filter(ImageFilter.GaussianBlur(2))

        img = np.array(img)
        blurred_data.append([img , ground_truth])

    
    return blurred_data

def mirror(data):
    mirrored_data = []
    for [image, ground_truth] in data:
        img = Image.fromarray(image)
        grdt = Image.fromarray(ground_truth)

        img = ImageOps.mirror(img)
        grdt = ImageOps.mirror(grdt)

        img = np.array(img)
        grdt = np.array(grdt)
        mirrored_data.append([img, grdt])

    return mirrored_data

def brightness(data):
    bright_data = []
    for [image, ground_truth] in data:
        img = Image.fromarray(image)
        img_darker = ImageEnhance.Brightness(img).enhance(0.5)
        img_brighter = ImageEnhance.Brightness(img).enhance(1.5)
        img_darker = np.array(img_darker)
        img_brighter = np.array(img_brighter)
        bright_data.append([img_darker, ground_truth])
        bright_data.append([img_brighter, ground_truth])
    return bright_data

def contrast(data):
    contrast_data = []
    for [image, ground_truth] in data:
        img = Image.fromarray(image)
        img_more = ImageEnhance.Contrast(img).enhance(0.5)
        img_less = ImageEnhance.Contrast(img).enhance(1.5)
        img_more = np.array(img_more)
        img_less = np.array(img_less)
        contrast_data.append([img_more, ground_truth])
        contrast_data.append([img_less, ground_truth])
    return contrast_data

# https://aparico.github.io/ & https://github.com/pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py
def pca(data):
    pca_data = []
    for [image, ground_truth] in data:
        orig_img = image.astype(float).copy()
        #normalize rgb values
        img = image/255.0
        #flatten to columns of rgb values
        img_rs = img.reshape(-1,3)
        #center each pixel around mean value
        img_centered = img_rs - np.mean(img_rs, axis=0)
        #3x3 covariance matrix
        img_cov = np.cov(img_centered, rowvar=False)

        eig_vals, eig_vecs = np.linalg.eigh(img_cov)
        sort_perm =  eig_vals[::-1].argsort()
        eig_vals[::1].sort()
        eig_vecs = eig_vecs[:, sort_perm]
        #[p1, p2, p3] eigvec matrix
        m1 = np.column_stack((eig_vecs))

        m2 = np.zeros((3,1))
        alpha = np.random.normal(0, 0.1)
        m2[:, 0] = alpha * eig_vals[:]
        add_vect = np.matrix(m1) * np.matrix(m2)

        for i in range(3):
            orig_img[..., i] += add_vect[i]
        orig_img = np.clip(orig_img, 0.0, 255.0)
        orig_img = orig_img.astype(np.uint8)
        pca_data.append([orig_img, ground_truth])

    return pca_data

def read_images(filenames , ground_truth_PATH, images_PATH):
    
    data = []
    for f in filenames:
        ground_truth = cv2.imread(join(ground_truth_PATH , f))
        image = cv2.imread(join(images_PATH , f))

        data.append([image , ground_truth])

    data = np.asarray(data)
    return data

def resize_images(dirs , base_dir , save = True):

    ctr = 0
    for file_name in dirs:
        ctr += 1
        print("\r " , ctr , "out of" , len(dirs) , end = "" , flush = True)
        img = Image.open(os.path.join(base_dir , file_name))
        img = img.resize((400, 400))

        if save:
            img.save(os.path.join(base_dir , file_name))
    
    print()

def main():

    filenames = [f for f in listdir(config.ground_truth_PATH_TRAIN) if isfile(join(config.ground_truth_PATH_TRAIN, f))]

    print("Reading files:")
    data = read_images(filenames , config.ground_truth_PATH_TRAIN , config.images_PATH_TRAIN)
    print("Read %d files of resolution %s"%(data.shape[0] , str(data[0].shape)))

    cv2.imshow("image",data[20][0])
    cv2.waitKey(0)
    cv2.imshow("ground truth",data[20][1])
    cv2.waitKey(0)

    print("Transforming data:")
    if config.ROTATE:
        print("rotating data")
        rotated_data = rotate(data)
        #cv2.imshow("rotated image" , rotated_data[0][0])
        #cv2.waitKey(0)
        #cv2.imshow("rotated ground truth" , rotated_data[0][1])
        #cv2.waitKey(0)

    if config.TRANSLATE:
        print("translating data")
        translated_data = translate(data)
        #cv2.imshow("translated image" , translated_data[0][0])
        #v2.waitKey(0)
        #cv2.imshow("translated ground truth" , translated_data[0][1])
        #cv2.waitKey(0)

    if config.ADD_NOISE: 
        print("Augmenting data with random Gaussian noise")
        noised_data = add_gaussian_noise(data)
        # cv2.imshow("noised image" , noised_data[0][0])
        # cv2.waitKey(0)
        # cv2.imshow("noised ground truth" , noised_data[0][1])
        # cv2.waitKey(0)
    
    if config.BLUR:
        print("blurring data")
        blurred_data = blur(data)
        # cv2.imshow("noised image" , blurred_data[0][0])
        # cv2.waitKey(0)
        #cv2.imshow("noised ground truth" , blurred_data[0][1])
        #cv2.waitKey(0)

    if config.MIRROR:
        print("mirroring data")
        mirrored_data = mirror(data)

    if config.BRIGHTNESS:
        print("brightness data")
        brightness_data = brightness(data)

    if config.CONTRAST:
        print("contrast data")
        contrast_data = contrast(data)

    if config.PCA:
        print("pca data")
        pca_data = pca(data)
    
    def removetree(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.removetree(path)  # remove the file
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove dir and all contains
        else:
            raise ValueError("file {} is not a file or dir.".format(path))

    def copytree(src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d) 
    
    print("starting to delete all augmented files...")
    if os.path.isdir(config.ground_truth_PATH_AUG):
        removetree(config.ground_truth_PATH_AUG)
    if os.path.isdir(config.images_PATH_AUG):
        removetree(config.images_PATH_AUG)

    print("finished deleting all augmented files...")

    print("copying everything here")
    copytree(config.base_DIR_training , config.base_DIR_aug)
    print("finished copying everything")

    size = len(listdir(config.images_PATH_TRAIN))

    ind = size+1
    #print(noised_data[0][0])
    
    #rotated data
    for j in range(len(rotated_data)):
        #images
            name = config.images_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(rotated_data[j][0])
            im.save(name)
    
        #groundtruth
            name = config.ground_truth_PATH_AUG+"/satImage" +"_" + str(ind)+".png"
            im = Image.fromarray(rotated_data[j][1])
            im.save(name)

            ind +=1
    #translated data
    for j in range(len(translated_data)):
        #images
            name = config.images_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(translated_data[j][0])
            im.save(name)
    
        #groundtruth
            name = config.ground_truth_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(translated_data[j][1])
            im.save(name)

            ind +=1
    #noised data
    for j in range(len(noised_data)):
        #images
            name = config.images_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray((noised_data[j][0]*255).astype(np.uint8))
            im.save(name)
    
        #groundtruth
            name = config.ground_truth_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(noised_data[j][1])
            im.save(name)

            ind +=1
    #blurred data
    for j in range(len(blurred_data)):
        #images
            name = config.images_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(blurred_data[j][0])
            im.save(name)
    
        #groundtruth
            name = config.ground_truth_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(blurred_data[j][1])
            im.save(name)

            ind +=1
    #mirrored data
    for j in range(len(mirrored_data)):
        #images
            name = config.images_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(mirrored_data[j][0])
            im.save(name)
    
        #groundtruth
            name = config.ground_truth_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(mirrored_data[j][1])
            im.save(name)

            ind +=1
    #brightness data
    for j in range(len(brightness_data)):
        #images
            name = config.images_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(brightness_data[j][0])
            im.save(name)
    
        #groundtruth
            name = config.ground_truth_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(brightness_data[j][1])
            im.save(name)

            ind +=1
    #contrast data
    for j in range(len(contrast_data)):
        #images
            name = config.images_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(contrast_data[j][0])
            im.save(name)
    
        #groundtruth
            name = config.ground_truth_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(contrast_data[j][1])
            im.save(name)

            ind +=1
    #pca data
    for j in range(len(pca_data)):
        #images
            name = config.images_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(pca_data[j][0])
            im.save(name)
    
        #groundtruth
            name = config.ground_truth_PATH_AUG+"/satImage"+"_" + str(ind)+".png"
            im = Image.fromarray(pca_data[j][1])
            im.save(name)

            ind +=1

def main_2():
    
    print("Resizing Images")
    dirs = os.listdir("Dataset/Train/Images/")
    resize_images(dirs , "Dataset/Train/Images/")

    print("Resizing Ground Truth")
    dirs = os.listdir("Dataset/Train/GroundTruth/")
    resize_images(dirs , "Dataset/Train/GroundTruth/")

if __name__ == "__main__":
    main_2()