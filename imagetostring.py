#!/usr/bin/python
# import necessary libraries
import sys
sys.path.append("/usr/local/lib/python3.5/site-packages")

import imutils
import numpy as np
import PIL

import cv2
import pytesseract

class ImageToString:
    def __init__(self, src_path="/home/pi/classification_yield/",
                 img_path="images/", log_path="logs/"):
        self.src_path = src_path
        self.img_path = img_path
        self.log_path = log_path
        
    def get_string(self, img_file):
        try:
            print("[INFO] loading image file...")
            img = cv2.imread(self.src_path + self.img_path + img_file)
            
            print("[INFO] converting to grayscale...")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
##            print("[INFO] manually selecting ROI...")
##            roi = cv2.selectROI(gray)
           
##            print("[INFO] cropping image...")
##            cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            
            if img_file == "xx.png":
                x = 1470
                y = 165
                roi = gray[y:y+45, x:x+40]
                roi = imutils.resize(roi, 195, 220)
                cv2.imshow("rohan", roi)
                cv2.waitKey(0)
                
            
            print("[INFO] applying gaussian blur...")
            blurred = cv2.GaussianBlur(roi, (11, 11), 0)
            
            print("[INFO] removing noise by dilation and erosion...")
            kernel = np.ones((1, 1), np.uint8)
            dilate = cv2.dilate(blurred, kernel, iterations=1)
            erode = cv2.erode(dilate, kernel, iterations=1)
            cv2.imwrite(self.src_path + self.img_path + "removed_noise.png", erode)
            
            print("[INFO] applying threshold to get black and white of image...")
            thresh = cv2.adaptiveThreshold(erode, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 2)
            
            print("[INFO] writing result to image file...")
            cv2.imwrite(self.src_path + self.img_path + "thresh.png", thresh)
            
            print("[INFO] recognizing text with tesseract...")
            result = pytesseract.image_to_string(PIL.Image.open(self.src_path + self.img_path + "thresh.png"),
                                                 config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789',
                                                 lang='eng')

            # DEBUG BGN
            print("[INFO] displaying output...")
            print("\n======\n" +
                  result +
                  "\n======\n")
            cv2.imshow("Output", thresh)
            cv2.waitKey(0)
            # DEBUG END
            
            print("[INFO] writing results to file...")
            f = open(self.src_path + self.log_path + img_file.split(".")[0] + ".log", "w")
            f.write(result)
            f.close

##            print("[INFO] removing template file...")
##            os.remove(temp)
            return result
        except Exception as e:
            f = open("/home/pi/classification_yield/logs/error.log", "w")
            f.write("imagetostring: " + str(datetime.now()) + " " + str(e))
            f.close
        

if __name__ == "__main__":
    ocr = ImageToString()
    ocr.get_string("xx.png")