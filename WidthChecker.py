import cv2; import numpy as np


def WidthCheck(frame):
    # define states for the line to be drawn
    drawing = False 
    startPoint = None  
    endPoint = None 
    width = []

    # define a mouse callback function that draws a line with mouse clicks
    def LineDrawing(event, x, y, flags, param):
        nonlocal drawing, startPoint, endPoint, width

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
            distance = np.linalg.norm(np.array(startPoint) - np.array(endPoint))*pixelCalibration
            width.append(distance)


    # set the windown name. create a window to select the ROI then bind the mouse callback to the window
    windowName = 'Width Check'
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 1980, 400)
    cv2.setMouseCallback(windowName, LineDrawing)

    # display the image and stop after hitting esc
    while True:
        lineImage = frame.copy()

        # if the points are made, draw the line
        if startPoint and endPoint:
            cv2.line(lineImage, startPoint, endPoint, (0, 255, 0), 3)

        # show the image. if you like the line, press esc
        cv2.imshow(windowName, lineImage)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # clear images
    cv2.destroyAllWindows()
    return width

pixelCalibration = np.load("Pixel Calibration.npy")

# variable inputs
folderName = 'Data/PVC P4 Test_7'
frame = cv2.imread(f'{folderName}/120.jpg')
width = WidthCheck(frame)
print(np.average(width))