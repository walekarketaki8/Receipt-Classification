# import the necessary packagess
from PIL import Image
import pytesseract
import os
import sys

IMAGE_PATH = sys.argv[1]
TEXT_PATH = sys.argv[2]

for file_ in (os.listdir(IMAGE_PATH)):
	print(file_[:-4])
	text = pytesseract.image_to_string(Image.open(os.path.join(IMAGE_PATH,file_)))
	mod_text = " ".join(text.split())
	file = open(os.path.join(TEXT_PATH,file_[:-4]+'.txt'),'w')
	file.write(mod_text)
	file.close() 
