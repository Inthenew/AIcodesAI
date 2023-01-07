import cv2
import numpy as np

# Set up camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame from camera
    _, frame = camera.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize previous frame to the same size as the current frame
    try:
        if prev_gray is None:
            prev_gray = np.zeros_like(gray)
    except Exception as err:
        prev_gray = np.zeros_like(gray)
    
    # Calculate difference between current frame and previous frame
    difference = cv2.absdiff(gray, prev_gray)
    
    # Threshold the difference to highlight changes in color
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    
    # Find contours in the difference image
    contours, hierarchy = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours are detected
    if len(contours) > 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Calculate the center of the contour
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            # Check if the contour is near the center of the frame
            if cx < frame.shape[1] * 0.3:
                # Move the machine to the left
                print('MOVE LEFT')
            elif cx > frame.shape[1] * 0.7:
                # Move the machine to the right
                print('MOVE RIGHT')
            else:
                # Check if the contour is within a certain distance from the center
                if cy < frame.shape[0] * 0.3:
                    # Turn left or right to avoid the object
                    print('MOVE RIGHT')
                else:
                    # Move the machine forward
                    print('MOVE FORWARD')
        else:
            # Move the machine forward
            print('MOVE FORWARD')
    
    # Update previous frame
    prev_gray = gray
    
    # Break loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up camera and close window
camera.release()
cv2.destroyAllWindows()
