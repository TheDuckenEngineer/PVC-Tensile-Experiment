import cv2; import numpy as np; import pandas as pd;
import trackpy as tp; import os ; import matplotlib.pyplot as plt
import time

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
        # mask = cv2.erode(mask, openingKernal)
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
    time.sleep(1)
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
        cv2.resizeWindow('Inspection', 960, 540)
        cv2.imshow('Inspection', frame)
        if cv2.waitKey(35) & 0xFF == 27:
            break
    
    # close the window previewer
    cv2.destroyAllWindows() 

    # define the tracking dataframe after reshaping it
    df = pd.DataFrame(np.array([df]).reshape((len(df), 4)), columns = ['y', 'x', 'frame', 'area'])

    # make a 3D plot of the marker area at its position
    fig = plt.figure(layout = 'constrained')
    ax = fig.add_subplot(projection = '3d')
    ax.set_xlabel('Length (pxl)')
    ax.set_ylabel('Width (pxl)')
    ax.set_title('Marker Size')
    ax.scatter3D(df.x, df.y, df.area, c = df.area)

    # show the areas per plane
    _, axs = plt.subplots(1, 2, layout = 'constrained')
    axs[0].scatter(df.y, df.area, c = df.area)
    axs[1].scatter(df.x, df.area, c = df.area)
    axs[0].set_xlabel('Width (pxl)', fontsize = 13)
    axs[1].set_xlabel('Length (pxl)', fontsize = 13)
    axs[0].set_ylabel('Marker Size', fontsize = 13)
    axs[0].set_title('Marker Size along Width')
    axs[1].set_title('Marker Size along Length')
    plt.show()
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
        tracked = tp.link(df, 15, memory = 15)
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
    # stress was recorded in MPa from the Instron
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
                                 "Stress (MPa)"])
    df["Axial Displacement (mm)"] = axDist    
    df["Axial Strain (pxl/pxl)"] = axStrain
    df["Transverse Displacement (mm)"] = transDist 
    df["Transverse Strain (pxl/pxl)"] = transStrain 
    df["Stress (MPa)"] = stress
    df.to_csv(f"Data\Processed data\{testName}.csv", sep = ',', header = True, index = False)
   
    return axDist, axStrain, transDist, transStrain, stress


def InstronDataReader(folder, filename):
    df = pd.read_csv(f'Data/Tensile Data/{folder}/{filename}')
    axDist = df["Axial Displacement (mm)"][1::].to_numpy()
    axStrain = df["Axial Strain (pxl/pxl)"][1::].to_numpy()
    transDist = df["Transverse Displacement (mm)"][1::].to_numpy()
    transStrain = df["Transverse Strain (pxl/pxl)"][1::].to_numpy()
    stress = df["Stress (MPa)"][1::].to_numpy()

    return axDist, axStrain, transDist, transStrain, stress


def InstronDataCompile(folder, plastiRatio):
    # import the processed data based on it's plasticizer content
    fileNames = [i for i in os.listdir(f'Data/Tensile Data/{folder}') if i.find(f'{plastiRatio}') != -1]
    
    # preallocate the total data vector 
    Data = np.zeros([0, 5])

    for i in fileNames:
        axDist, axStrain, transDist, transStrain, stress = InstronDataReader(folder, i)
        data = np.vstack([axDist, axStrain, transDist, transStrain, stress]).T
        Data = np.vstack([Data, data])
    Data = Data[Data[::, 1].argsort()]
    axDist = Data[::, 0]
    axStrain = Data[::, 1]
    transDist = Data[::, 2]
    transStrain = Data[::, 3]
    stress = Data[::, 4]

    return axDist, axStrain, transDist, transStrain, stress



def ViscoelasticDataProcessor(folderName, name):
    # import the excel file. dont use columns beyond 4 since they're empty  due to 
    # needing a place to put extra comments
    df = pd.read_excel(f"Data/Viscoelastic Data/{folderName}/{name}", header = None, usecols = [0, 1, 2, 3, 4])

    # get sample geometric data. length is in mm and area is in mm^2
    sampleLength = df.loc[5][1]*1e-3 # sample length - converted from mm to m
    sampleArea = df.loc[6][1]*1e-6 # sample area - converted from mm^2 to m^2

    # preallocate the measurement names
    data = df.loc[df.index[28::]].to_numpy(dtype = float)
    
    # dataColumns = ['Time (s)', 'Temp. (Cel)', 'Displacement (m)', 'Load (N)', 'Displacement Rate (m/s)']
    expTime = data[:, 0]*60      # time -  converted from min to sec
    strain = data[:, 2]*1e-6/sampleLength    # displacement - converted from um to m
    stress = data[:, 3]*1e-6/sampleArea     # force - converted from uN to N
    strainRate = data[:, 4]*1e-6/60/sampleLength    # displacement rate - converted from um/min to m/s then to strain/s
    return expTime, strain, strainRate, stress

def ViscoelasticDataViewer(plastiRatio):
    # define the folder name and pull the files that end with xlsx
    folderName = 'Monotonic Strain Rate Data'
    fileNames = [i for i in os.listdir(f'Data/Viscoelastic Data/{folderName}') if i.endswith('.xlsx') and i.find(plastiRatio) != -1]

    # plot parameters
    markerSize = 0.5
    titleSize = 15
    axisSize = 11
    legendSize = 9

    for i in fileNames:
        # read and process the data file for strain and stress 
        _, strain, _, stress = ViscoelasticDataProcessor(folderName, i)

        # find each strain amplitude region
        regions = MonotonicStrainRateRegionSelector(strain)
        for j in range(0, 3):
            indexRange = range(regions[0, j], regions[1, j])
            strainFit = strain[indexRange] - strain[indexRange][0]
            stressFit = stress[indexRange] - stress[indexRange[0]]
            plt.scatter(strainFit, stressFit, s = markerSize, label = f' Trial: {i.removesuffix('.xlsx').split('_')[1]} - Region {j}')
    plt.xlabel('Strain (m/m)', fontsize = axisSize)
    plt.ylabel('Stress (Pa)', fontsize = axisSize)
    plt.title(f'{i.removesuffix('.xlsx').split('_')[0]} Fitting Region', fontsize = titleSize)
    plt.legend(fontsize = legendSize)
    plt.show()


    # define the folder name and pull the files that end with xlsx
    folderName = 'Stress Relaxation Data'
    fileNames = [i for i in os.listdir(f'Data/Viscoelastic Data/{folderName}') if i.endswith('.xlsx') and i.find(plastiRatio) != -1]

    for i in fileNames:
        # read and process the data file for strain and stress 
        expTime, strain, _, stress = ViscoelasticDataProcessor(folderName, i)

        # the starting point is 4 before the maximum stress value
        regions = StressRelaxationRegionSelector(expTime, stress)

        for j in range(0, 3): 
            # extract the increasing strain regions. since the start is 5 indices from the maximum stress,
            # use these points to get the offset strain and stress
            indexRange = range(regions[0, j], regions[1, j])
            
            # get the stress vector offset
            expTimeOffset = expTime[indexRange][0]
            stressOffset = stress[indexRange][0]
            strainOffset = strain[indexRange][0]
            
            # reset the index range so the first data point is the maximum stress value.
            indexRange =  range(regions[0, j] + 5, regions[1, j])

            # define the variables 
            expTimeFit = expTime[indexRange] - expTimeOffset
            strainFit = strain[indexRange] - strainOffset
            stressFit = stress[indexRange] - stressOffset
            normStressFit = stressFit/strainFit
            plt.scatter(expTimeFit, normStressFit, s = markerSize + 2, label = f' Trial: {i.removesuffix('.xlsx').split('_')[1]} - Region {j}')

    plt.xlabel('Time (s)', fontsize = axisSize)
    plt.ylabel('Stress (Pa)/Strain (m/m)', fontsize = axisSize)
    plt.title(f'{fileNames[0].removesuffix('.xlsx').split('_')[0]}', fontsize = titleSize)
    plt.legend(fontsize = legendSize)
    plt.show()

def MonotonicStrainRateRegionSelector(strain):
    # find the indices wher strain has a large change
    regionIndices = np.where(np.diff(strain) > 1e-3)[0]

    # find where the index changes more than 1. this is used to segment each region
    startIndices = regionIndices[np.where(np.diff(regionIndices) > 1)[0] + 1]
    startIndices = np.insert(startIndices, 0, regionIndices[0])
    endIndices  = regionIndices[np.where(np.diff(regionIndices) > 1)[0]]
    endIndices = np.insert(endIndices, 2, regionIndices[-1])

    regions = np.array([startIndices, endIndices])    
    return regions

def StressRelaxationRegionSelector(expTime, stress):

    # preallocate the maximum stress index
    maxIndex = np.zeros([0], dtype = int)

    # find the maximum stress for each region
    for i in range(0, 3):
        # find between time values
        regionIndex = np.where(np.logical_and(expTime > i*3400, expTime < (i+1)*3400))[0]

        # find the index of the maximum value and concate to the maximum stress index
        maxIndex = np.hstack([maxIndex, regionIndex[np.where(stress[regionIndex] == np.max(stress[regionIndex]))[0]]])

    # make the start index 5 before the maximum stress
    startIndices =  np.array(maxIndex) - 5

    # get the end indices based on the points before the start of the next region
    endIndices = np.array([startIndices[1] - 50, startIndices[2] - 50, len(stress) - 130])
    regions = np.array([startIndices, endIndices])
    return regions

