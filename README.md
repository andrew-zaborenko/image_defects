# image_defects

A repository for creating small defects on grayscale images. The code also features slight augmentations (rotation, color change, pad and crop). The method `distort_image` return the altered images and the list of bouding boxes in `imgaug` and yolo formats.
All the functions can be found in `image_defects_script.py`. Notebook `image_defects_imgaug.ipynb` contains visualizations of the code.
Sample code:
```
img = cv2.imread('logo.jpg',0)
mask = ...
img_dist, bb_list, yolo_bbs = distort_image(img, mask)
```

Mask is an array with the same shape as the image, with 1's denoting the logo on the image and 0's denoting th e background. It can be loaded from an external source (it's just a numpy array) or can be naively generated in the following manner:

```
PATH_TO_IMG = 'logo.jpg'
MASK_THRESHOLD = 25

img = cv2.imread(PATH_TO_IMG, 0)
mask = np.full(img.shape, fill_value=0.0, dtype=np.float32)
for x in range(img.shape[1]):
    for y in range(img.shape[0]):
        if img[y, x] <= MASK_THRESHOLD:
            mask[y, x] = 1
```
`MASK_THRESHOLD` is the color threshold to destinguish between the black logo and white background. 
More examples and visualizations can be found in the notebook.

Sample output:
![image](https://user-images.githubusercontent.com/72259060/200409800-2a27c23e-71e5-4773-bf84-67ed23f575e8.png)
