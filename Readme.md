# Bone_X-Ray-Dompetition
Identification of forehand is broken or not from a X-ray radiographs automatically with better than human performance using deep learning 
methods ( Designnet ).

### MURA Dataset
MURA is a dataset of musculoskeletal radiographs consisting of 14,982 studies from 12,251 patients, with a total of 40,895 multi-view radiographic images. Each study belongs to one of seven standard upper extremity radiographic study types: elbow, finger, forearm, hand, humerus, shoulder and wrist.

### Requirements

```sh
 cv2
 keras
 numpy
 python
```

### For training Run...

```sh
$ python train.py
```

### For Predictions...
Copy images to predict folder and run the following:
```sh
$ python interface.py
```
