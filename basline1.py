from sklearn import linear_model
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import config
"""
    To execute the code run the following command
        python baseline1.py

"""

if __name__ == '__main__':
    # Helper functions

    def load_image(infilename):
        data = mpimg.imread(infilename)
        return data

    def img_float_to_uint8(img):
        rimg = img - np.min(img)
        rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
        return rimg

    # Concatenate an image and its groundtruth
    def concatenate_images(img, gt_img):
        nChannels = len(gt_img.shape)
        w = gt_img.shape[0]
        h = gt_img.shape[1]
        if nChannels == 3:
            cimg = np.concatenate((img, gt_img), axis=1)
        else:
            gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
            gt_img8 = img_float_to_uint8(gt_img)
            gt_img_3c[:,:,0] = gt_img8
            gt_img_3c[:,:,1] = gt_img8
            gt_img_3c[:,:,2] = gt_img8
            img8 = img_float_to_uint8(img)
            cimg = np.concatenate((img8, gt_img_3c), axis=1)
        return cimg

    def img_crop(im, w, h):
        list_patches = []
        imgwidth = im.shape[0]
        imgheight = im.shape[1]
        is_2d = len(im.shape) < 3
        for i in range(0,imgheight,h):
            for j in range(0,imgwidth,w):
                if is_2d:
                    im_patch = im[j:j+w, i:i+h]
                else:
                    im_patch = im[j:j+w, i:i+h, :]
                list_patches.append(im_patch)
        return list_patches


    # Loaded a set of images
    
    

    #image_dir = os.path.join(data_dir, "training/training/images")
    #test_files_dir = os.path.join(data_dir, 'test_images/test_images')
    #test_files = os.listdir(test_files_dir)
    files = os.listdir(config.ground_truth_PATH_TRAIN)
    n = len(files) # Load maximum 20 images
    print("Loading " + str(n) + " images")
    imgs = [load_image(os.path.join(config.images_PATH_TRAIN, files[i])) for i in range(n) if files[i].endswith('.png')]
    #print(files[0])

    #gt_dir = os.path.join(data_dir,  ground_truth_PATH_TRAIN)
    print("Loading " + str(n) + " images")
    gt_imgs = [load_image(os.path.join(config.ground_truth_PATH_TRAIN, files[i])) for i in range(n) if files[i].endswith('.png')]
    print(files[0])

    n = 100 # Only use 10 images for training


    #print('Image size = ' + str(imgs[0].shape[0]) + ',' + str(imgs[0].shape[1]))

    # Show first image and its groundtruth image
    cimg = concatenate_images(imgs[0], gt_imgs[0])
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(cimg, cmap='Greys_r')


    # Extract patches from input images
    patch_size = 16 # each patch is 16*16 pixels

    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

    #print("img_patches: ", np.array(img_patches).shape )
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    #print("img_patches: ", img_patches.shape)
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    # Extract 6-dimensional features consisting of average RGB color as well as variance
    def extract_features(img):
        feat_m = np.mean(img, axis=(0,1))
        feat_v = np.var(img, axis=(0,1))
        feat = np.append(feat_m, feat_v)
        return feat

    # Extract 2-dimensional features consisting of average gray color as well as variance
    def extract_features_2d(img):
        #print("small image dim: ", img.shape)
        feat_m = np.mean(img)
        feat_v = np.var(img)
        feat = np.append(feat_m, feat_v)
        return feat

    # Extract features for a given image
    def extract_img_features(filename):
        img = load_image(filename)
        #print("img size: ", img.shape)
        img_patches = img_crop(img, patch_size, patch_size)
        X = np.asarray([extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
        return X


    # Compute features for each image patch
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

    def value_to_class(v):
        df = np.sum(v)
        #print("df: ", df)
        if df > foreground_threshold:
            return 1
        else:
            return 0

    X = np.asarray([extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    #print("the shape of X: ", X.shape)
    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    # Print feature statistics

    print('Computed ' + str(X.shape[0]) + ' features')
    print('Feature dimension = ' + str(X.shape[1]))
    print('Number of classes = ' + str(np.max(Y)))

    print("enumerate: ", enumerate(Y))
    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print('Class 0: ' + str(len(Y0)) + ' samples')
    print('Class 1: ' + str(len(Y1)) + ' samples')

    # Display a patch that belongs to the foreground class
    plt.imshow(gt_patches[Y1[3]], cmap='Greys_r')

    # Plot 2d features using groundtruth to color the datapoints
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

    # train a logistic regression classifier

    # we create an instance of the classifier and fit the data
    logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
    print("testing...", " X size: ", X.shape, " Y size: ", Y.shape)
    logreg.fit(X, Y)

    # Predict on the training set
    Z = logreg.predict(X)

    # Get non-zeros in prediction and grountruth arrays
    Zn = np.nonzero(Z)[0]
    Yn = np.nonzero(Y)[0]

    TPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
    print('True positive rate = ' + str(TPR))

    # Plot features using predictions to color datapoints
    plt.scatter(X[:, 0], X[:, 1], c=Z, edgecolors='k', cmap=plt.cm.Paired)


    # Convert array of labels to an image

    def label_to_img(imgwidth, imgheight, w, h, labels):
        im = np.zeros([imgwidth, imgheight])
        idx = 0
        for i in range(0, imgheight, h):
            for j in range(0, imgwidth, w):
                im[j:j + w, i:i + h] = labels[idx]
                idx = idx + 1
        return im


    def make_img_overlay(img, predicted_img):
        w = img.shape[0]
        h = img.shape[1]
        color_mask = np.zeros((w, h, 3), dtype=np.uint8)
        color_mask[:, :, 0] = predicted_img * 255

        img8 = img_float_to_uint8(img)
        background = Image.fromarray(img8, 'RGB').convert("RGBA")
        overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
        new_img = Image.blend(background, overlay, 0.2)
        return new_img

    def prediction_to_file(idimg, res, submission_file):
        predictions = res.reshape(38,38)
        for j in range(0, predictions.shape[1]):
            for i in range(0, predictions.shape[0]):
                with open(submission_file, 'a') as f:
                    _id = '{:03d}'.format(idimg)
                    f.write(f'{_id}_{j*patch_size}_{i*patch_size},{predictions[i,j]}\n')


    submission_file = 'submission_baseline1.csv'
    try:
        os.remove(submission_file)
    except OSError:
        pass

    with open(submission_file, 'a') as f:
        f.write('id,prediction\n')

    # Run prediction on the img_idx-th image
    ids = [int(filename.split('_')[1].split('.')[0]) for filename in os.listdir(config.base_DIR_testing)]
    #print("ids: ", ids)
    ids.sort()

    print("predicting...")
    for id in ids:
    #for path in files:
        Xtest = extract_img_features(f'{config.base_DIR_testing}/test_{id}.png')
        res = logreg.predict(Xtest)
        #print("the shape of Xtest: ", Xtest.shape)
        #print("the shape of res: ", res.shape)
        plt.scatter(Xtest[:, 0], Xtest[:, 1], c=res, edgecolors='k', cmap=plt.cm.Paired)
        prediction_to_file(id, res, submission_file)
