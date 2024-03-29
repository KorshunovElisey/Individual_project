import cv2
import pytesseract as pts
import cv2
import operator
import numpy as np
import time
from numba import jit

pts.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

time_n = time.time()

class frameExtractor:
    def __init__(self):
        self.alpha = 3  # Contrast control (1.0-3.0)
        self.beta = 50  # Brightness control (0-100)
        self.blur_kernel_size = 5
        self.thresh_block_size = 51
        self.thresh_offset = 10
        self.pts_model = "ssd"  # "ssd", "letsgodigital" or "7seg"
        self.psm = 8
        self.oem = 1
        self.erosion_kernel_size = 3
        self.erosion_iterations = 1
        self.denoise_h = 10
        self.denoise_template_window_size = 7
        self.denoise_search_window_size = 21
        self.record = 1
        
    
    def pre_process_image(self, img, dilate=False):
        proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
        proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        proc = cv2.bitwise_not(proc, proc)
        if dilate:
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
            proc = cv2.dilate(proc, kernel)
        return proc
    
    def find_corners_of_largest_polygon(self, img):
        contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        polygon = contours[0]  # Largest image
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
    
    def distance(self, p1, p2):
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a**2) + (b**2))


    def crop(self, img, crop_rect):
        top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
        # crop the image
        left = min(top_left[0], bottom_left[0])
        right = max(top_right[0], bottom_right[0])
        top = min(top_left[1], top_right[1])
        bottom = max(bottom_left[1], bottom_right[1])
        return img[top:bottom, left:right]
    
    def toFixed(numObj, digits, etc):
        numObj = str(digits)
        new_res = []
        for i in range(len(numObj)):
            if len(numObj)-i == etc: 
                new_res.append('.')
            new_res.append(numObj[i])
        new_res = ''.join(new_res)
        return new_res
    
    def final_prediction(self, resize_factor, margin_x, margin_y, img, dst_folder_name, digits_for_res=0):
        if img is None:
            img = cv2.imread(self.src_file_name, cv2.IMREAD_GRAYSCALE)
        proc = self.pre_process_image(img, dilate=True)
        # corners = self.find_corners_of_largest_polygon(proc)
        # img = self.crop(img, corners)
        # cv2.imwrite(dst_folder_name + "/1cropped.png", img)
        img = img[margin_x : img.shape[0] - margin_x, margin_y : img.shape[1] - margin_y]
        cv2.imwrite(dst_folder_name + "/2cropped.png", img)
        ## brightness and contrast
        img = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
        cv2.imwrite(dst_folder_name + "/3contrast.png", img)
        # smoothing
        img = cv2.GaussianBlur(img, (self.blur_kernel_size, self.blur_kernel_size), 0)
        cv2.imwrite(dst_folder_name + "/4blur.png", img)
        # denoise
        img = cv2.fastNlMeansDenoising(
            img,
            None,
            h=self.denoise_h,
            templateWindowSize=self.denoise_template_window_size,
            searchWindowSize=self.denoise_search_window_size,
        )
        cv2.imwrite(dst_folder_name + "/5denoise.png", img)
        # thresholding
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.thresh_block_size,
            self.thresh_offset,
        )
        cv2.imwrite(dst_folder_name + "/6threshold.png", img)
        # erode
        kernel = np.ones((self.erosion_kernel_size, self.erosion_kernel_size), np.uint8)
        img = cv2.erode(img, kernel, iterations=self.erosion_iterations)
        cv2.imwrite(dst_folder_name + "/7erode.png", img)
        # dilate
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        img = cv2.dilate(img, kernel)
        cv2.imwrite("test/8dilate.png", img)
        img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        cv2.imwrite(dst_folder_name + "/9downscale.png", img) 
        res = pts.image_to_string(
            img,
            lang=self.pts_model,
            config=f"--psm {self.psm} --oem {self.oem} -c tessedit_char_whitelist=.0123456789",
        )

        final_res = []
        for sub in res:
            final_res.append(sub.replace("\n", ""))
        
        final_res = ''.join(final_res)
        final_res = final_res.replace(".", "")
        if final_res == '':
           final_res = 0
        final_res = self.toFixed(final_res, digits_for_res)
        with open(dst_folder_name + "/result.txt", 'a') as file:
            time_now = time.time()
            file.write(str(time_now) + '\t' + str(final_res) + '\n' )

        return(img, final_res)


# resize_factor = 0.15
# margin_x = 0
# margin_y = 0

# image_processor = frameExtractor(dst_folder_name='F:\PyhonScripts\Individual_project\kort')
# cv_vid = cv2.VideoCapture(0)

# while True:
#     print(time.time()-time_n)
#     time_n = time.time()
#     ret, cv_img = cv_vid.read()
#     img_grey = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
#     processed_image = image_processor.final_prediction(resize_factor=resize_factor, margin_x=margin_x, margin_y=margin_y, img=img_grey, dst_folder_name='F:\PyhonScripts\Individual_project\kort', digits_for_res=0)
#     cv2.imshow('d', processed_image)
#     k = cv2.waitKey(30) & 0xFF
#     if k == 27:
#         break

