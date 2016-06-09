from PIL import Image
import os
import random
import numpy as np

def get_immediate_subdirectories(a_dir):
    # returns a list of folders at the path of the argument
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def classifier_folder_to_jpg_dict_list(folder_of_categories,reject_folder_path):
    # arguments are a main folder containing subfolders, one of each
    # class, filled with images to feed the classifier.
    # Images in the class subfolders can be '.JPG', '.jpg', '.jpeg' or '.png'
    # all '.png' images will be converted to '.jpg', those files that have a
    # different extension will be moved to the reject_folder_path, make
    # sure the paths end with a foward slash '/'

    # function returns a python dictionary called one_hot_labels which has
    # as keys the number of classes ie. 1,2,3 4 ....k, and as keys the name of
    # each class subfolder in alphabetical order, example:
    # {1: 'High_Quality', 2: 'Medium_Quality', 3: 'Unable_Poor'}

    # function returns a python list called category_list which is a
    # list of lists, there is a list for each class in alphabetical order
    # each list in the main list contains names of each image in that class

    # function from the fxn.py file
    main_folders = get_immediate_subdirectories(folder_of_categories)

    num_cat = len(main_folders)

    jpegs = ['.JPG', '.jpg', '.jpeg']

    # we will fill these two
    one_hot_labels = {}
    category_list = []

    if not os.path.exists(reject_folder_path):
        os.mkdir(reject_folder_path)

    i = 1
    for cat_folder in main_folders:
        one_hot_labels[i] = cat_folder
        i+=1
        cat_folderpath = folder_of_categories + cat_folder

        for name in os.listdir(cat_folderpath):

            filename, file_extension = os.path.splitext(cat_folderpath+ '/' + name)
            #print(file_extension,type(file_extension))
            if file_extension not in jpegs:
                if file_extension == '.png':
                    im = Image.open(cat_folderpath+ '/' + name)
                    im.save(filename + ".jpg")
                    os.remove(cat_folderpath+ '/' + name)

                else:
                    os.rename(cat_folderpath+ '/' + name, reject_folder_path + name)

        category_list.append([filename for filename in os.listdir(cat_folderpath)
                             if os.path.isfile(os.path.join(cat_folderpath, filename))])

    return  one_hot_labels, category_list

def cat_list_to_list_of_tuples(img_dict,cat_list):
    #turns a list of lists into a list of tuples
    num_classes = len(cat_list)
    identity = np.identity(num_classes)

    list_of_tuples = []
    i = 0
    for class_list in cat_list:
        for name in class_list:
            list_of_tuples.append( [identity[i], img_dict[i+1] + '/' + name] )
            #list_of_tuples.append( (i, img_dict[i+1] + '/' + name) )
        i+=1

    return list_of_tuples
