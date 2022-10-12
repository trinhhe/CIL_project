from os import listdir
from os.path import isfile, join

base_DIR_training = "Data/training/training/"
base_DIR_testing = "Data/test_images/test_images/"
base_DIR_aug = "Data/training_augmented/training_augmented/"

ground_truth_PATH_TRAIN = join(base_DIR_training , "groundtruth") 
images_PATH_TRAIN = join(base_DIR_training , "images")


ground_truth_PATH_AUG = join(base_DIR_aug , "groundtruth") 
images_PATH_AUG = join(base_DIR_aug, "images")

ROTATE = True
TRANSLATE = True
ADD_NOISE = True
BLUR = True
MIRROR = True
BRIGHTNESS = True
CONTRAST = True
PCA = True
