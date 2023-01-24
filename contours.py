import pytesseract as pts
import matplotlib.pyplot as plt

pts.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = plt.imread('img/test.jpg')

config = rf"--psm 6 --oem 0 -c tessedit_char_whitelist=.0123456789"
res = pts.image_to_string(img, lang="letsgodigital", config=config)
print(res)