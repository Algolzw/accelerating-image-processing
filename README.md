# accelerating-image-processing
Advanced Scientific Python Programming (ASPP 2024) course project

### Description
Current researches in computer vision always require a huge number of high-quality images as base datasets. Before training machine learning models, an important step is to preprocess these high-quality and high-resolution images into small patches and add some noises and blur artifacts to improve the data diversity. However, most preprocessing operations are time-consuming on large size images and sometimes we even need 2-3 days to apply them to large datasets on a remote server. 

In this project, I'd like to accelerate the image preprocessing on a large dataset by using Python (with the *multiprocessing* package) and hopefully we can largely reduce the image preprocessing times.


### Code Useage

1. Put all high-quality images to the `images` directory.
2. Run degradation with a single process with `python generate_LQ_images.py`, you can also repeat the degradation process by specifying the number in the argument: `python generate_LQ_images.py 100`.
3.  Run multi-process degradation with multiple CPUs with `python sync_generate_LQ_images.py`, you can also repeat the degradation process by specifying the number in the argument: `sync_generate_LQ_images.py 100`.
4. The degraded images will be saved in the `output` directory.

I tried this code on a Mac M1 computer (8 cores), the single process code takes ~68s for 100 degradations, while the multi-process code only takes ~14s, 5 times faster! And if you run the code on a large server, you can use more cpus to speed up large-scale LQ image generations.