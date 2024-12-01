import cv2; import numpy as np; import pandas as pd;
import trackpy as tp; import os


def Mask(folderName, i, kernel):
    # read the first and last images 
    frame = cv2.imread(f'{folderName}/{i}')

    # blur the image a bit. convert color to hsv from rgb.
    # blur = cv2.GaussianBlur(frame, (3,3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # select the blue color range which is 95 to 140 hue. openCV uses hue up to 180
    upperLim1 = np.array([140, 255, 255])
    lowerLim1 = np.array([80, 90, 105])
    mask = cv2.inRange(hsv, lowerLim1, upperLim1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, (7,7))

    # find the contours from the blue mask
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return frame, contours


def MarkerIdentify(folderName, small, big, openingKernal):
    # list the image names from the directory and separate the data file and images
    fileName = os.listdir(folderName)
    imageNames =  [fileName for fileName in fileName if fileName.endswith('jpg')]

    for i in ['0.jpg', f'{len(imageNames)-2}.jpg']:
        
        # apply the mask to the frame
        frame, contours = Mask(folderName, i, openingKernal)

        # find the centroid of the contours and save to data frame
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (area > small) & (area < big):
                print(area)
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.drawContours(frame, cnt, -1, (0, 225, 225), 3)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

        # show overlayed contours and the frame number
        cv2.namedWindow('Inspection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Inspection', 960, 200)
        cv2.imshow('Inspection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  
    return 


def ParticleIdentify(folderName, small, big, openingKernal):
    # list the image names from the directory and separate the data file and images
    fileName = os.listdir(folderName)
    imageNames =  [fileName for fileName in fileName if fileName.endswith('jpg')]  
        
    # preallocate dataframe array
    df = []

    for i in imageNames:
        # apply the mask to the frame
        _, contours = Mask(folderName, i, openingKernal)
        
        # find the centroid of the contours and save to data frame
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (area > small) & (area < big):
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                df.append([cy, cx, i.rstrip('.jpg'), area])

    # define the tracking dataframe after reshaping it
    df = pd.DataFrame(np.array([df]).reshape((len(df), 4)), columns = ['y', 'x', 'frame', 'area'])

    # run particle linking through trackpy. use a movement of 5 pixels and use 10
    # frames to remember non-existing pixels
    try:
        tracked = tp.link(df, 12, memory = 13)
        tracked.drop(columns = ['area'], inplace = True)
    except IndexError:
        print('No particles identified')
    return tracked


def CosineCorrection(folderName, ax0, ax1):
    # define a psudo length vector. after finding the specimen angle, the length between markers
    # will be used as our adjance length for the specimen angle to be compared. this sudo length
    # vector is a constant length opposite to the specimen angle that we're solving for. it's an
    # estimation to back solve for the specimen angle.
    
    psudoY = []
    startPosition = []
    cosFrame = [1000, 2000, 4000, 6000, 8000]
    for j in cosFrame:
        # import the image, shaprne it, read it
        image = cv2.imread(f'{folderName}/{j}.jpg')
        img = image[::, range(ax1[0,j], ax0[0,j])]

        # define states for the line to be drawn
        drawing = False 
        startPoint = None  
        endPoint = None 

        # define a mouse callback function that draws a line with mouse clicks
        def LineDrawing(event, x, y, flags, param):
            nonlocal drawing, startPoint, endPoint

            if event == cv2.EVENT_LBUTTONDOWN:  # start drawing line
                drawing = True
                startPoint = (x, y)
                endPoint = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:  # update the line as the mouse moves
                if drawing:
                    endPoint = (x, y)

            elif event == cv2.EVENT_LBUTTONUP:  # finalize the line
                drawing = False
                endPoint = (x, y)

        # set the windown name. create a window to select the ROI then bind the mouse callback to the window
        windowName = 'Cosine correction'
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 1980, 400)
        cv2.setMouseCallback(windowName, LineDrawing)

        # display the image and stop after hitting esc
        while True:
            lineImage = img.copy()

            # if the points are made, draw the line
            if startPoint and endPoint:
                cv2.line(lineImage, startPoint, endPoint, (0, 255, 0), 1)

            # show the image. if you like the line, press esc
            cv2.imshow(windowName, lineImage)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # clear images
        cv2.destroyAllWindows()

        # determine the angle error of the area. create a psudo opposite length based on the current 
        # marker x axis separation. this is necessary to estimate the area length change.
        specimenAngle = np.arctan(np.abs((endPoint[1] - startPoint[1])/(endPoint[0] - startPoint[0])))
        psudoY.append((ax0[0,j] - ax1[0,j])*np.tan(specimenAngle))
        startPosition.append(startPoint)

    # calculate the average oppoisite length. 
    psudoY  = np.mean(psudoY)

    # determine the axial points angle offset.
    axialAngle = [np.arctan(np.abs((ax0[1,i] - ax1[1,i])/(ax0[0,i] - ax1[0,i]))) for i in range(0, len(ax0.T))]

    # determine the speciemn centerline angle.
    specimenCenterLine = [np.arctan(np.abs((psudoY)/(ax0[0,i] - ax1[0,i]))) for i in range(0, len(ax0.T))]

    # determine the axial angle offset from centerline
    axialAngleOffset = np.abs(np.array([axialAngle[i] - specimenCenterLine[i] for i in range(0, len(axialAngle))]))

    # plot the centerline, showing correctness of the centerline
    for i in cosFrame:
        img = cv2.imread(f'{folderName}/{i}.jpg')

        # set the windown name. create a window to select the ROI then bind the mouse callback to the window
        windowName = f'Cosine correction Frame: {i}'
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 1980, 300)

        # set the points for the line to be drawn
        adjLength = np.abs(ax0[0,i] - ax1[0,i])
        newPosition = (ax0[0,i] - adjLength, ax0[1,i] - int(adjLength*np.tan(specimenCenterLine)[i]) - 40)
        origPosition = (ax0[0,i], ax0[1,i] - 40)

        # draw across up from the axial centroids
        cv2.line(img, origPosition, newPosition, (0, 0, 255), 2)
        cv2.imshow(windowName, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    check = input('Is centerline accurate? (enter/n)')
    if check =='':
        return specimenCenterLine, axialAngleOffset
    elif check == 'n':
        return CosineCorrection(folderName, ax0, ax1)


def StrainFunction(folderName, objects):
    # copy the data frame so it doesn't get destroyed
    # objects = tracked.copy()

    # list the image names from the directory and separate the data file and images
    testName = folderName.split('/')[1]
    fileName = os.listdir(folderName)
    dataName =  [fileName for fileName in fileName if fileName.endswith('csv')][0]

    # all particles should exist in the same frame. remove frames that don't appear 4 times. 
    objects.drop(index = objects[objects.particle > 3].index, inplace = True)
 
    # list the particles. if there is more than 4, remove them. find the indices for 
    # axial particles using the max and min x-position.  
    position = [objects.x[i] for i in objects.particle.unique()]
    axParticles = [position.index(max(position)), position.index(min(position))]
    transParticles = [i for i in objects.particle.unique() if i not in axParticles]
    pair = [axParticles, transParticles] 

    # import stress data from the excel sheet but only choose data points that exist in the frames
    data = pd.read_csv(f'{folderName}/{dataName}')['Tensile stress'][1::].to_numpy(dtype = float)

    # find instanaces where image frames exist and save these isntances of stress data
    stressLen = len(data)
    frameMax = min([len(objects[objects['particle'] == i]) for i in range(0, 4)])
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
    
    # perform angular corrections
    specimenCenterLine, axialAngleOffset = CosineCorrection(folderName, ax0, ax1)

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
    df["Specimen Angle (Rad)"] = specimenCenterLine
    df["Axial Angle Offset (Rad)"] = axialAngleOffset
    df.to_csv(f"Processed data\{testName}.csv", sep = ',', header = True, index = False)
   
    return axDist, axStrain, transDist, transStrain, stress, specimenCenterLine, axialAngleOffset


def DataReader(filename):
    df = pd.read_csv(f'Processed data/{filename}')
    axDist = df["Axial Displacement (mm)"][1::].to_numpy()
    axStrain = df["Axial Strain (pxl/pxl)"][1::].to_numpy()
    transDist = df["Transverse Displacement (mm)"][1::].to_numpy()
    transStrain = df["Transverse Strain (pxl/pxl)"][1::].to_numpy()
    stress = df["Stress (MPa)"][1::].to_numpy()
    specimenCenterLine = df["Specimen Angle (Rad)"][1::].to_numpy()
    axialAngleOffset = df["Axial Angle Offset (Rad)"][1::].to_numpy()

    return axDist, axStrain, transDist, transStrain, stress, specimenCenterLine, axialAngleOffset


def DataComplile(plastiRatio):
    # import the processed data based on it's plasticizer content
    fileNames = [i for i in os.listdir('Processed data') if i.find(f'{plastiRatio}') != -1]
    
    # preallocate the total data vector 
    Data = np.zeros([0, 7])

    for i in fileNames:
        axDist, axStrain, transDist, transStrain, stress, specimenCenterLine, axialAngleOffset = DataReader(i)
        data = np.vstack([axDist, axStrain, transDist, transStrain, stress, specimenCenterLine, axialAngleOffset]).T
        Data = np.vstack([Data, data])
    axDist = Data[::, 0]
    axStrain = Data[::, 1]
    transDist = Data[::, 2]
    transStrain = Data[::, 3]
    stress = Data[::, 4]
    specimenCenterLine = Data[::, 5]
    axialAngleOffset = Data[::, 6]

    return axDist, axStrain, transDist, transStrain, stress, specimenCenterLine, axialAngleOffset