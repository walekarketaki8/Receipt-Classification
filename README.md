                                        Receipt Classification Using Machine Learning
ML Pipeline:
Dataset Collection(Positive + Negative) --> Image to text conversion --> Processing on text --> Converting processed text into numerical data --> Train classifier --> Test classifier(Input is image)

Dataset Collection-

* Dataset scraped through internet. Wrote a code using "selenium-webdriver" python package to download images using query. Downloaded image count is 882 (546(receipt images) + 336(not receipt images)). Gathered both positive and negative samples in one folder(IMAGE_DATA which is not included in folder because of memory concerns) with positive examples names changes to 'POS_' + name of image and negative examples names changes to 'NEG_'+name of image which will help in labelling.
    Example use: python image_download.py 'query name' numbers-of-images-to-download

