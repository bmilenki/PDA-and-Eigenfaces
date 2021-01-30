import numpy as np
import os
import cv2
from PIL import Image
from scipy import stats
from matplotlib import pyplot as plt



def main():
    Question1()
    facesMatrix1600, featureMeans, featureStdDevs, eigenValues, eigenVectors = Question2()
    Question3(facesMatrix1600, featureMeans, featureStdDevs, eigenValues, eigenVectors)


def Question1():
    # Starting part C

    print("Question 1 Start:----------------------------------------")

    X_org = np.array([[0, 1], [0, 0], [1, 1], [0, 0], [1, 1], [1, 0], [1, 0], [1, 1], [2, 0], [2, 1]])

    # In class example of
    # X_org = np.array([[4,1,2],[2,4,0],[2,3,-8],[3,6,0],[4,4,0],[9,10,1],[6,8,-2],[9,5,1],[8,7,10],[10,8,-5]])
    # print(X_org)
    Y = np.array([[1], [1], [1], [1], [1], [0], [0], [0], [0], [0]])
    # print(Y)

    # splitting so we can zscore based on column
    f1, f2 = np.hsplit(X_org, 2)

    # using scipy to z score, ddof makes us use n-1
    f1_zscored = stats.zscore(f1, ddof=1)
    f2_zscored = stats.zscore(f2, ddof=1)

    # X is now z-scored by column/feature
    X = np.concatenate((f1_zscored, f2_zscored), 1)

    print("Z Scored X: \n", X)

    covX = np.cov(np.transpose(X), ddof=1)

    print("covX: \n", covX)

    U, S, V = np.linalg.svd(covX)

    eigenValues = S
    eigenVectors = U

    print("EigenValues:\n", eigenValues)

    print("EigenVectors:\n", eigenVectors)

    # Starting part E
    ev1, ev2 = np.hsplit(eigenVectors, 2)

    newX = np.matmul(X_org, ev1)
    print("newX:\n", newX)

    print("Question 1 End:----------------------------------------")

def Question2():
    print("Question 2 Start:----------------------------------------")
    facesNameList = []
    facesMatrix1600 = []

    featureMeans = []
    featureStdDevs = []

    # creating directory link and reading in all files
    facesFileDir = "yalefaces"
    allFileNames = os.listdir(facesFileDir)

    # saving all files into a list
    for fName in allFileNames:
        if fName == "Readme.txt":
            pass
        else:
            facesNameList.append(facesFileDir + "\\" + fName)

    # opening every file in the list, resizing them, flattening them, and saving to final list
    for facesName in facesNameList:
        image = Image.open(facesName)
        imageResized = image.resize((40, 40))
        imageResizedData = np.array(imageResized)
        imageResizedData = imageResizedData.astype("float64")
        flattenImageData = imageResizedData.flatten(order="C")

        facesMatrix1600.append(flattenImageData)

    facesMatrix1600 = np.array(facesMatrix1600)

    # array to standardize
    for i in range(len(facesMatrix1600[0])):  # loops 1600 times for each pixel
        currPixel = []

        # need to get the mean and std dev for each pixel
        for j in range(len(facesMatrix1600)):  # loops 154 times for each image
            currPixel.append(facesMatrix1600[j][i])

        meanCurrPixel = np.mean(currPixel)
        stdDevCurrPixel = np.std(currPixel, ddof=1)

        featureMeans.append(meanCurrPixel)
        featureStdDevs.append(stdDevCurrPixel)

        # now we can standardize
        for k in range(len(facesMatrix1600)):  # loops for each image
            facesMatrix1600[k][i] = (facesMatrix1600[k][i] - meanCurrPixel) / stdDevCurrPixel

    # reducing the data using PCA
    covFaces = np.cov(np.transpose(facesMatrix1600), ddof=1)
    U, S, V = np.linalg.svd(covFaces)

    eigenValues = S
    eigenVectors = V

    # uncomment to see eigenvalues
    # print("EigenValues:\n", eigenValues)

    # print("EigenVectors:\n", eigenVectors)

    # the two largest are just the first two since the function orders it
    eVal1 = eigenValues[0]
    eVal2 = eigenValues[1]

    eVec1 = np.array(eigenVectors[0])
    eVec2 = np.array(eigenVectors[1])

    # uncomment to check norms
    # print(np.linalg.norm(eVec1))
    # print(np.linalg.norm(eVec2))

    W = np.array([eVec1, eVec2]).transpose()

    Z = np.matmul(facesMatrix1600, W)

    newX, newY = np.hsplit(Z, 2)

    plt.scatter(x=newX, y=newY, facecolors='none', edgecolors="blue")
    # plt.gca().invert_yaxis() # can we used to flip so we get the same image as in the pdf
    plt.show()

    print("Question 2 End:----------------------------------------")
    # returns the values needed for Q3
    return facesMatrix1600, featureMeans, featureStdDevs, eigenValues, eigenVectors

def Question3(facesMatrix1600, featureMeans, featureStdDevs, eigenValues, eigenVectors):
    print("Question 3 Start:----------------------------------------")
    img_array = []

    person = facesMatrix1600[0]  # gets subject02.centerlight

    for k in range(1, len(person)+1):
        W = eigenVectors[0:k]
        # transposing them to columns
        W = np.array(W).transpose()

        # getting our Z
        Z = np.matmul(person, W)

        newX = np.matmul(Z, W.transpose())

        # undoes standardization
        for i in range(len(newX)):
            newVal = (newX[i] * featureStdDevs[i])+featureMeans[i]
            # so the pixel values dont go out of range and become black
            if newVal > 255:
                newVal = 255
            if newVal < 0:
                newVal = 0
            newX[i] = newVal

        # unflatten the picture
        newPicData = np.reshape(newX, (40, 40))
        newPicData = newPicData.astype(np.uint8)

        img_array.append(newPicData)


    height, width = img_array[0].shape
    size = (width, height)


    out = cv2.VideoWriter('Milenki_Assignment1.avi', cv2.VideoWriter_fourcc(*"DIVX"), 60, size)

    for j in range(len(img_array)):
        currImage = cv2.cvtColor(img_array[j], cv2.COLOR_GRAY2BGR)
        out.write(currImage)
    out.release()

    print("Question 3 End:----------------------------------------")

if __name__ == "__main__":
    main()
