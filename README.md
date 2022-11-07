# image_defects

A repository for creating small defects on grayscale images. The code also features slight augmentations (rotation, color change, pad and crop). The method `distort_image` return the altered images and the list of bouding boxes in `imgaug` and yolo formats.

Sample code:
```
img = cv2.imread('logo.jpg',0)
mask = ...
img_dist, bb_list, yolo_bbs = distort_image(img, mask)
```

Mask is an array with the same shape as the image, with 1's denoting the logo on the image and 0's denoting th e background. It can be loaded from an external source (it's just a numpy array) or can be naively generated in the following manner:

```

```
