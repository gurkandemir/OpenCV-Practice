import numpy as np
import cv2

image_name = 'nature-travel.jpg'

# Reading image
img = cv2.imread(image_name, cv2.IMREAD_COLOR)

# Put text onto image
cv2.putText(img, "Gurkan Demir", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)

# Draw a white line
cv2.line(img, (0, 0), (100, 100), (255, 255, 255), 25)

# Draw a green circle
cv2.circle(img, (250, 90), 60, (0, 255, 0), -1)

# Draw a yellow rectangle
cv2.rectangle(img, (700, 150), (1250, 630), (0, 255, 255), 10)

# Draw poly-lined red rectangle
pts = np.array([[500, 70], [200,300], [570, 590], [650, 370]], np.int32)
cv2.polylines(img, [pts], True, (0, 0, 255), 3)

cv2.imshow('image', img)
cv2.waitKey(0)
