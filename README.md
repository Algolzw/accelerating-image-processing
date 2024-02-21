# accelerating-image-processing
Advanced Scientific Python Programming (ASPP 2024) course project

### Description
Current researches in computer vision always require a huge number of high-quality images as base datasets. Before training machine learning models, an important step is to preprocess these high-quality and high-resolution images into small patches and add some noises and blur artifacts to improve the data diversity. However, most preprocessing operations are time-consuming on large size images and sometimes we even need 2-3 days to apply them to large datasets on a remote server. 

In this project, I'd like to accelerate the image preprocessing on a large dataset by using Python (with the *multiprocessing* package) and hopefully we can largely reduce the image preprocessing times.
