import cv2

image_name = "inputs/nature-travel.jpg"

# Reads image
img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

# Shows original image
cv2.imshow("Original image", img)
cv2.waitKey(0)

# Shows resized image
resized_img = cv2.resize(img, (0, 0), fx = 0.75, fy = 0.75)
cv2.imshow("Resized image", resized_img)
cv2.waitKey(0)
