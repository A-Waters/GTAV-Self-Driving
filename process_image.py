import cv2

def process_image(image):
    image_colored = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_normal = cv2.normalize(image_colored, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return cv2.resize(image_normal, (80,60)) 

