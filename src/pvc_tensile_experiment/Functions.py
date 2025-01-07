import cv2; import numpy as np; import pandas as pd;
import trackpy as tp; import os ; import plotly.express as px


def Mask(frame, lowerColorLims, useKernel, kernelSize):
    # convert color to hsv from rgb.
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # select the blue color range which is 95 to 140 hue. openCV uses hue up to 180
    upperLim = np.array([140, 255, 255])
    lowerLim = np.array(lowerColorLims)
    mask = cv2.inRange(hsv, lowerLim, upperLim)
    
    # apply a mask if desired
    if useKernel == True:
        openingKernal = np.ones((kernelSize, kernelSize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, openingKernal)

    # find the contours from the blue mask
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def MaskCheck(folderName, searchRegion, lowerColorLims, useKernel, kernelSize):
    # list the image names from the directory and separate the data file and images
    fileName = os.listdir(folderName)
    imageNames =  [fileName for fileName in fileName if fileName.endswith('jpg')]  

    # preallocate dataframe array
    df = []

    for i in range(0, len(imageNames), 50):
        
        # pull the frame from the folder
        frame = cv2.imread(f'{folderName}/{i}.jpg')

        # find the contours from the blue mask
        contours = Mask(frame, lowerColorLims, useKernel, kernelSize)
        
        # find the centroid of the contours and save to data frame
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (area > searchRegion[0]) & (area < searchRegion[1]):
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                df.append([cy, cx, i, area])

                # draw contours and their centers
                cv2.drawContours(frame, cnt, -1, (100, 255, 255), 1)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # setp and display the contours and centers
        cv2.namedWindow('Inspection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Inspection', 1080, 300)
        cv2.imshow('Inspection', frame)
        cv2.waitKey(35)
    
    # close the window previewer
    cv2.destroyAllWindows() 

    # define the tracking dataframe after reshaping it
    df = pd.DataFrame(np.array([df]).reshape((len(df), 4)), columns = ['y', 'x', 'frame', 'area'])

    # plot the centroids as they evolve 
    fig = px.scatter_3d(x = df.x, y = df.y, z = df.area)
    fig.show()
    return


def ParticleIdentify(folderName, searchRegion, lowerColorLims, useKernel, kernelSize):
    # list the image names from the directory and separate the data file and images
    fileName = os.listdir(folderName)
    imageNames =  [fileName for fileName in fileName if fileName.endswith('jpg')]  
        
    # preallocate dataframe array
    df = []

    for i in range(0, len(imageNames), 1):

        # pull the frame from the folder
        frame = cv2.imread(f'{folderName}/{i}.jpg')

        # apply the mask to the frame
        contours = Mask(frame, lowerColorLims, useKernel, kernelSize)
        
        # find the centroid of the contours and save to data frame
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (area > searchRegion[0]) & (area < searchRegion[1]):
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                df.append([cy, cx, i, area])

    # define the tracking dataframe after reshaping it
    df = pd.DataFrame(np.array([df]).reshape((len(df), 4)), columns = ['y', 'x', 'frame', 'area'])

    # run particle linking through trackpy. use a movement of 10 pixels and use 12
    # frames to remember non-existing pixels
    try:
        tracked = tp.link(df, 10, memory = 12)
        tracked.drop(columns = ['area'], inplace = True)
    except IndexError:
        print('No particles identified')
    return tracked


def StrainFunction(folderName, objects):
    # copy the data frame so it doesn't get destroyed
    # objects = tracked.copy()

    # list the image names from the directory and separate the data file and images
    testName = folderName.split('/')[1]
    fileName = os.listdir(folderName)
    dataName =  [fileName for fileName in fileName if fileName.endswith('csv')][0]

    # all particles should exist in the same frame. remove frames that don't appear 4 times. 
    objects.drop(index = objects[objects.particle > 3].index, inplace = True)
    objects.reset_index(inplace=True)

    # list the particles. if there is more than 4, remove them. find the indices for 
    # axial particles using the max and min x-position.  
    position = [objects.x[i] for i in objects.particle.unique()]
    axParticles = [position.index(max(position)), position.index(min(position))]
    transParticles = [int(i) for i in objects.particle.unique() if i not in axParticles]
    pair = [axParticles, transParticles] 

    # import stress data from the excel sheet but only choose data points that exist in the frames
    data = pd.read_csv(f'{folderName}/{dataName}')['Tensile stress'][1::].to_numpy(dtype = float)

    # find instanaces where image frames exist and save these isntances of stress data
    stressLen = len(data)
    frameMax = min([len(objects[objects['particle'] == i]) for i in np.arange(0, 4)])
    if stressLen < frameMax:
        stress = data
        frameRange = range(0, stressLen)
    elif stressLen > frameMax:
        frameRange = range(0, frameMax)
        stress = np.array(data[frameRange])
    else:
        stress = data

    # convert the x/y axial and transverse particle to numpy arrays
    ax0XPos = objects[objects['particle'] == pair[0][0]].x.to_numpy(dtype = int)[frameRange]
    ax0YPos = objects[objects['particle'] == pair[0][0]].y.to_numpy(dtype = int)[frameRange]
    ax1XPos = objects[objects['particle'] == pair[0][1]].x.to_numpy(dtype = int)[frameRange]
    ax1YPos = objects[objects['particle'] == pair[0][1]].y.to_numpy(dtype = int)[frameRange]
    trans0XPos = objects[objects['particle'] == pair[1][0]].x.to_numpy(dtype = int)[frameRange]
    trans0YPos = objects[objects['particle'] == pair[1][0]].y.to_numpy(dtype = int)[frameRange]
    trans1XPos = objects[objects['particle'] == pair[1][1]].x.to_numpy(dtype = int)[frameRange]
    trans1YPos = objects[objects['particle'] == pair[1][1]].y.to_numpy(dtype = int)[frameRange]

    # store the x and y positions per particle in an 2D - row array 
    ax0 = np.vstack([ax0XPos, ax0YPos])
    ax1 = np.vstack([ax1XPos, ax1YPos])
    trans0 = np.vstack([trans0XPos, trans0YPos])
    trans1 = np.vstack([trans1XPos, trans1YPos])

    # define ararys to put distance between axial and transverse markers
    axDist = np.zeros(0)
    transDist = np.zeros(0)

    # calculate the eculidean distance between the particles
    for i in range(0, len(ax0XPos)):
        axDist = np.hstack([axDist, np.linalg.norm(ax0[::, i] - ax1[::, i])])
        transDist = np.hstack([transDist, np.linalg.norm(trans0[::, i] - trans1[::, i])])

    # calculate engineering strains 
    axStrain = abs(axDist - axDist[0])/axDist[0]
    transStrain = abs(transDist - transDist[0])/transDist[0]
    
    # export the data 
    df = pd.DataFrame(columns = ["Axial Displacement (mm)", "Axial Strain (pxl/pxl)", 
                                 "Transverse Displacement (mm)", "Transverse Strain (pxl/pxl)", 
                                 "Stress", 
                                 "Specimen Angle", "Axial Angle Offset"])
    df["Axial Displacement (mm)"] = axDist    
    df["Axial Strain (pxl/pxl)"] = axStrain
    df["Transverse Displacement (mm)"] = transDist 
    df["Transverse Strain (pxl/pxl)"] = transStrain 
    df["Stress (MPa)"] = stress
    df.to_csv(f"Processed data\{testName}.csv", sep = ',', header = True, index = False)
   
    return axDist, axStrain, transDist, transStrain, stress


def DataReader(filename):
    df = pd.read_csv(f'Processed data/{filename}')
    axDist = df["Axial Displacement (mm)"][1::].to_numpy()
    axStrain = df["Axial Strain (pxl/pxl)"][1::].to_numpy()
    transDist = df["Transverse Displacement (mm)"][1::].to_numpy()
    transStrain = df["Transverse Strain (pxl/pxl)"][1::].to_numpy()
    stress = df["Stress (MPa)"][1::].to_numpy()

    return axDist, axStrain, transDist, transStrain, stress


def DataComplile(plastiRatio):
    # import the processed data based on it's plasticizer content
    fileNames = [i for i in os.listdir('Processed data') if i.find(f'{plastiRatio}') != -1]
    
    # preallocate the total data vector 
    Data = np.zeros([0, 5])

    for i in fileNames:
        axDist, axStrain, transDist, transStrain, stress = DataReader(i)
        data = np.vstack([axDist, axStrain, transDist, transStrain, stress]).T
        Data = np.vstack([Data, data])
    axDist = Data[::, 0]
    axStrain = Data[::, 1]
    transDist = Data[::, 2]
    transStrain = Data[::, 3]
    stress = Data[::, 4]

    return axDist, axStrain, transDist, transStrain, stress