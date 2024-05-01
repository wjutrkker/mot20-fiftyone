import os
import pandas as pd
import shutil
import cv2

# Creating lists of file folders for each train set. 
fileList = ["/data/MOT20/train/MOT20-01/", "/data/MOT20/train/MOT20-02/", "/data/MOT20/train/MOT20-03/", "/data/MOT20/train/MOT20-05/"]
columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y"]

yoloIndex=0
# Run through each folder. 
for partPath in fileList:
    path = os.path.join(partPath, "gt/gt.txt")
    # create paths as necessary. 
    imgPath = os.path.join(partPath, "img1")
    newImgPath = os.path.join(partPath, "images")
    labelPath = os.path.join(partPath, "labels")
    shutil.rmtree(labelPath)
    if not os.path.isdir(labelPath):
        os.makedirs(labelPath, exist_ok=True)
    if os.path.isdir(imgPath):
        shutil.move(imgPath, newImgPath)
    seqinfoPath = os.path.join(partPath, "seqinfo.ini")
    with open(seqinfoPath,"r") as f: 
        seqinfo = f.readlines()
    # pull the camera info for the relative bounding boxes 
    print(seqinfo) 
    for item in seqinfo:
        if "imWidth" in item:
            width = int(str(item).replace("\n","").replace("imWidth=", ""))
        if "imHeight" in item:
            height = int(str(item).replace("\n","").replace("imHeight=", ""))
        if "seqLength" in item:
            numImages = int(str(item).replace("\n","").replace("seqLength=", ""))

    # read the det.txt file. 
    data = pd.read_csv(path, sep=",", header=None)
    data.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y"]

    # Write out the detection data to the yolov5 files. 
    for index, row in data.iterrows():
        img1 = str(int(row['frame'])).rjust(6, '0')
        index = f"{str(yoloIndex+int(img1)).rjust(6, '0')}"
        imgnew = f"{index}.jpg"
        labelnew = f"{index}.txt"
        newlabelname = os.path.join(labelPath, labelnew )
        aw = float(row["bb_width"])/width
        ah = float(row["bb_height"])/height
        cy = float(row["bb_top"] + row["bb_height"]/2)/height
        cx = float(row["bb_left"] + row["bb_width"]/2)/width
        yoloLine = f"{int(row['id'])} {cx} {cy} {aw} {ah} \n"