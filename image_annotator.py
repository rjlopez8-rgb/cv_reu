import cv2

original_image = cv2.imread("Jup03.jpg")

image = cv2.resize(original_image, (0, 0), fx=0.5, fy=0.5)

# border
cv2.rectangle(
    image,
    (10, 10),
    (image.shape[1]-10, image.shape[0]-10),
    (255, 0, 0),
    5
)

# caption background
cv2.rectangle(
    image,
    (20, 20),
    (350, 80),
    (0, 0, 0),
    -1
)

# caption
cv2.putText(
    image,
    "Jupiter Observation",
    (30, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.imshow("Annotated", image)
cv2.waitKey(0)
cv2.destroyAllWindows()