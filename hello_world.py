print ("Hello world")
import cv2

image = cv2.imread("Jup03.jpg")

cv2.imshow("Jupiter", image)

cv2.waitKey(0)
cv2.destroyAllWindows()