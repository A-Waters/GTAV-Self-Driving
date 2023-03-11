import cv2

def process_image(image):
    image_colored = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_normal = image_colored/255
    return cv2.resize(image_normal, (480,270)) 

