import tkinter as tk
import math
import numpy as np
import statsmodels.api as sm
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.measure import regionprops
from skimage import io
from skimage.filters import threshold_otsu  # For finding the threshold for grayscale to binary conversion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from tkinter import filedialog


def rgbgrey(img):
    # Converts rgb to grayscale
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg


def greybin(img):
    # Converts grayscale to binary
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)  # to remove small components or noise
    #     img = ndimage.binary_erosion(img).astype(img.dtype)
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg


def preproc(path, img=None, display=True):
    if img is None:
        img = mpimg.imread(path)
        img = mpimg.imread(path)
        img = mpimg.imread(path)
        img = mpimg.imread(path)
    if display:
        plt.imshow(img)
        plt.show()
    grey = rgbgrey(img)  # rgb to grey
    if display:
        plt.imshow(grey, cmap=matplotlib.cm.Greys_r)
        plt.show()
    binimg = greybin(grey)  # grey to binary
    if display:
        plt.imshow(binimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    r, c = np.where(binimg == 1)
    # Now we will make a bounding box with the boundary as the position of pixels on extreme.
    # Thus we will get a cropped image with only the signature part.
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    return signimg


def Ratio(img):
    a = 0
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == True:
                a = a + 1
    total = img.shape[0] * img.shape[1]
    return a / total


def Centroid(img):
    numOfWhites = 0
    a = np.array([0, 0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == True:
                b = np.array([row, col])
                a = np.add(a, b)
                numOfWhites += 1
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a / numOfWhites
    centroid = centroid / rowcols
    return centroid[0], centroid[1]


def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return r[0].solidity


def SkewKurtosis(img):
    h, w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value
    # calculate projections along the x and y axes
    xp = np.sum(img, axis=0)
    yp = np.sum(img, axis=1)
    # centroid
    cx = np.sum(x * xp) / np.sum(xp)
    cy = np.sum(y * yp) / np.sum(yp)
    # standard deviation
    x2 = (x - cx) ** 2
    y2 = (y - cy) ** 2
    sx = np.sqrt(np.sum(x2 * xp) / np.sum(img))
    sy = np.sqrt(np.sum(y2 * yp) / np.sum(img))

    # skewness
    x3 = (x - cx) ** 3
    y3 = (y - cy) ** 3
    skewx = np.sum(xp * x3) / (np.sum(img) * sx ** 3)
    skewy = np.sum(yp * y3) / (np.sum(img) * sy ** 3)

    # Kurtosis
    x4 = (x - cx) ** 4
    y4 = (y - cy) ** 4
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp * x4) / (np.sum(img) * sx ** 4) - 3
    kurty = np.sum(yp * y4) / (np.sum(img) * sy ** 4) - 3

    return (skewx, skewy), (kurtx, kurty)


def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    img = preproc(path, display=display)
    ratio = Ratio(img)
    centroid = Centroid(img)
    solidity = EccentricitySolidity(img)
    skewness, kurtosis = SkewKurtosis(img)
    retVal = (ratio, centroid, solidity, skewness, kurtosis)
    return retVal


def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (1, temp[0], temp[1][0], temp[1][1], temp[2], temp[3][0], temp[3][1], temp[4][0], temp[4][1])
    return features


def makeCSV(genuine_image_paths, forged_image_paths):
    if not (os.path.exists('C:\\Users\\heman\\PycharmProjects\\practice\\Images\\Features')):
        os.mkdir('C:\\Users\\heman\\PycharmProjects\\practice\\Images\\Features')
        print('New folder "Features" created')
    if not (os.path.exists('C:\\Users\\heman\\PycharmProjects\\practice\\Images\\Features/Training')):
        os.mkdir('C:\\Users\\heman\\PycharmProjects\\practice\\Images\\Features/Training')
        print('New folder "Features/Training" created')
    if not (os.path.exists('C:\\Users\\heman\\PycharmProjects\\practice\\Images\\Features/Testing')):
        os.mkdir('C:\\Users\\heman\\PycharmProjects\\practice\\Images\\Features/Testing')
        print('New folder "Features/Testing" created')
    gpath = genuine_image_paths
    # forged signatures path
    fpath = forged_image_paths
    # for person in range(1,2):
    # per = ('00'+str(person))[-3:]
    # print('Saving features for person id-',per)

    with open('C:\\Users\\heman\\PycharmProjects\\practice\\Images\\Features\\Training/training.csv', 'w') as handle:
        handle.write('constant,ratio,cent_y,cent_x,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
        # Training set
        for i in range(0, 4):
            source = os.path.join(gpath, str(i) + '.png')
            features = getCSVFeatures(path=source)
            handle.write(','.join(map(str, features)) + ',1\n')
        for i in range(0, 4):
            source = os.path.join(fpath, str(i) + '.png')
            features = getCSVFeatures(path=source)
            handle.write(','.join(map(str, features)) + ',0\n')

    with open('C:\\Users\\heman\\PycharmProjects\\practice\\Images\\Features\\Testing/testing.csv', 'w') as handle:
        handle.write('constant,ratio,cent_y,cent_x,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
        # Testing set
        for i in range(3, 5):
            source = os.path.join(gpath, str(i) + '.png')
            features = getCSVFeatures(path=source)
            handle.write(','.join(map(str, features)) + ',1\n')
        for i in range(3, 5):
            source = os.path.join(fpath, str(i) + '.png')
            features = getCSVFeatures(path=source)
            handle.write(','.join(map(str, features)) + ',0\n')


def testing(path):
    feature = getCSVFeatures(path)
    if not (os.path.exists('C:\\Users\\heman\\PycharmProjects\\practice\\Images/TestFeatures')):
        os.mkdir('C:\\Users\\heman\\PycharmProjects\\practice\\Images/TestFeatures')
    with open('C:\\Users\\heman\\PycharmProjects\\practice\\Images\\TestFeatures/testcsv.csv', 'w') as handle:
        handle.write('constant,ratio,cent_y,cent_x,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature)) + '\n')


def sigmoid(z=np.array([])):
    e = format(1 / (1 + math.exp(100)), '.2f')
    return e


# makeCSV()
def mlearn(test_image_path):
    n_input = 9
    # train_person_id = input("Enter person's id : ")
    # test_image_path = "E:\\Axis dataset\\Hemant\\real/1.png"
    train_path = 'C:\\Users\\heman\\PycharmProjects\\practice\\Images\\Features\\Training/training.csv'
    testing(test_image_path)
    test_path = 'C:\\Users\\heman\\PycharmProjects\\practice\\Images\\TestFeatures/testcsv.csv'

    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    model = LogisticRegression(C=50, solver='liblinear', random_state=0).fit(train_input, correct)
    z = model.predict_proba(test_input)
    if z[0][1] < 0.5:
        return 0;
    else:
        return 1;


a = []
b = []
c = []


def browsefunc_1():
    filename = filedialog.askopenfilename(filetypes=(("All files", "*.*"), ("tiff files", "*.tiff")))
    ent1.insert(tk.END, filename)  # add this
    a.append(filename)


def browsefunc_2():
    filename = filedialog.askopenfilename(filetypes=(("All files", "*.*"), ("tiff files", "*.tiff")))
    ent2.insert(tk.END, filename)  # add this
    b.append(filename)


def browsefunc_3():
    filename = filedialog.askopenfilename(filetypes=(("All files", "*.*"), ("tiff files", "*.tiff")))
    ent3.insert(tk.END, filename)  # add this
    c.append(filename)


def perfect_path(path):
    i = 0
    a = 0
    while a == 0:
        if path[-i] == '/':
            a = 1
        i = i + 1
    return path[:-(i - 1)]


def submit():
    name = name_entry.get()
    real_path = perfect_path(a[-1])
    fake_path = perfect_path(b[-1])
    test_path = c[-1]
    makeCSV(real_path, fake_path)
    q = mlearn(test_path)
    if q == 0:
        submit_label = tk.Label(root, text="The Signature do not Belongs to a Valid person",
                                font=('calibre', 10, 'bold'))
        submit_label.grid(row=6, column=1)
    else:
        submit_label = tk.Label(root, text="The Signature Belongs to a Valid person", font=('calibre', 10, 'bold'))
        submit_label.grid(row=6, column=1)


root = tk.Tk()
root.title("SIGNATURE AUTHENTICATION")
root.geometry("400x400")
name_ = tk.StringVar()
# name
name_label = tk.Label(root, text='Username', font=('calibre', 10, 'bold'))
name_entry = tk.Entry(root, textvariable=name_, font=('calibre', 10, 'normal'))
name_label.grid(row=1, column=0)
name_entry.grid(row=1, column=1)

# Real dataset  entry
br_1 = tk.Label(root, text='Real dataset', font=('calibre', 10, 'bold'))
br_1.grid(row=2, column=0)
ent1 = tk.Entry(root, font=40)
ent1.grid(row=2, column=1)
b1 = tk.Button(root, text="Browse", font=40, command=browsefunc_1)
b1.grid(row=2, column=3)

# fake dataset entry
br_2 = tk.Label(root, text='Fake dataset', font=('calibre', 10, 'bold'))
br_2.grid(row=3, column=0)
ent2 = tk.Entry(root, font=40)
ent2.grid(row=3, column=1)
b2 = tk.Button(root, text="Browse", font=40, command=browsefunc_2)
b2.grid(row=3, column=3)

# test dataset
br_3 = tk.Label(root, text='Test dataset', font=('calibre', 10, 'bold'))
br_3.grid(row=4, column=0)
ent3 = tk.Entry(root, font=40)
ent3.grid(row=4, column=1)
b3 = tk.Button(root, text="Browse", font=40, command=browsefunc_3)
b3.grid(row=4, column=3)

check = tk.Button(root, text="check", width=10, command=submit)
check.grid(row=5, column=1)
root.mainloop()
