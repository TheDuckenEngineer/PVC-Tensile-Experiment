import cv2; import numpy as np; import pandas as pd;
import trackpy as tp; import os

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

