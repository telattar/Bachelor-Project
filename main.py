from ultralytics import YOLO
import cv2
from super_resolution import cartoon_upsampling_4x
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, Image, ImageDraw, ImageFilter
import numpy as np
import os
import easyocr
import math
import csv
from datetime import datetime

model = YOLO("runs/detect/train18/weights/best.pt")
reader = easyocr.Reader(['ar'])
vid = cv2.VideoCapture(0)
centerPrev = []
endpoints = []
# object tracking
tracked = {}
# comparing tracked object ocr results
trackedCompareStrings = {}
# tracking the objects that no longer need ocr
trackedDone = {}
trackid = 0
count = 0
outText = ''
arabicAlpha = 'أابتثجحخدذرزسشصضطظعغفقكلمنهوى'
arabicNums = '١٢٣٤٥٦٧٨٩'

ocrFlag = False


def corrector(textRecog):
    textRecog = textRecog.replace('ذ', 'د')
    textRecog = textRecog.replace('ا', 'أ')
    textRecog = textRecog.replace('ت', 'ب')
    textRecog = textRecog.replace('ث', 'ب')
    textRecog = textRecog.replace('خ', 'ج')
    textRecog = textRecog.replace('ح', 'ج')
    textRecog = textRecog.replace('ز', 'ر')
    textRecog = textRecog.replace('ش', 'س')
    textRecog = textRecog.replace('ض', 'ص')
    textRecog = textRecog.replace('ظ', 'ط')
    textRecog = textRecog.replace('غ', 'ع')
    textRecog = textRecog.replace('9', 'و')
    # sometimes the letter ein is seen as a number four
    for i in range(len(textRecog)):
        if textRecog[i] == '٤':
            if i + 1 != len(textRecog):
                if textRecog[i + 1] in arabicAlpha:
                    textRecog = textRecog[:i] + 'ع' + textRecog[i + 1:]
    return textRecog


# skew adjustment
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def compute_skew(src_img):
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 4.0, maxLineGap=h / 4.0)
    angle = 0.0

    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt) * 180 / math.pi


def all_same(items):
    return all(x == items[0] for x in items)

def tracker():
    # first time tracking
    global trackid
    if tracked == {}:
        for pt in centerCurrent:
            for pt2 in centerPrev:
                dist = abs(pt[0][0] - pt2[0][0])
                if dist < 120:
                    tracked[trackid] = pt
                    trackedCompareStrings[trackid] = []
                    trackedDone[trackid] = False
                    trackid += 1
                    continue

    elif trackid != 0:
        trackingCopy = tracked.copy()
        centerCopy = centerCurrent.copy()
        for objid, pt2 in trackingCopy.items():
            exist = False
            for pt in centerCopy:
                dist = abs(pt[0][0] - pt2[0][0])
                # keep tracking the object if the distance is less than 10
                if dist < 120:
                    tracked[objid] = pt
                    exist = True
                    if pt in centerCurrent:
                        centerCurrent.remove(pt)
                    continue

            if not exist:
                tracked.pop(objid)
                trackedDone.pop(objid)
                trackedCompareStrings.pop(objid)

        for pt in centerCurrent:
            tracked[trackid] = pt
            trackedCompareStrings[trackid] = []
            trackedDone[trackid] = False
            trackid += 1

def imageProc_ocr(frame):
    global outText
    # check each tracked object
    trackingCopy = tracked.copy()
    for objid in trackingCopy.keys():
        if trackedDone.get(objid):
            continue
        if not trackedDone.get(objid):
            start = tracked.get(objid)[0]
            end = tracked.get(objid)[1]
            cropTop = (end[1] - start[1]) * 0.35
            yStart = start[1] + cropTop

            cropped = frame[int(yStart):end[1] - 1, start[0]: end[0]]

            if cropped.std() < 45:
                cropped = cv2.convertScaleAbs(cropped, alpha=1.5, beta=30)

            elif cropped.std() > 60:
                cropped = cv2.convertScaleAbs(cropped, alpha=0.5, beta=0)

            filename = 'frame' + str(count) + '.jpg'
            cv2.imwrite(filename, cropped)
            upsampledFN = str(filename) + 'ups.jpg'
            superResolution = cartoon_upsampling_4x(filename, upsampledFN)
            img = cv2.imread(upsampledFN)
            img = Image.fromarray(img)
            img = img.filter(ImageFilter.SHARPEN)
            img = np.array(img)
            # skew adjust
            # img = rotate_image(img, compute_skew(img))
            upsampledFNS = "upsampledfinal" + str(count) + '.jpg'
            cv2.imwrite(upsampledFNS, img)
            OCR = reader.readtext(upsampledFNS, detail=0)
            print(OCR)
            if len(OCR) > 1:
                outText = OCR[0] + OCR[1]
            elif len(OCR) == 1:
                outText = OCR[0]

            outText = outText.replace(" ", "")
            outText = corrector(outText)
            updated = trackedCompareStrings.get(objid)
            updated.append(outText)
            trackedCompareStrings[objid] = updated
            print(trackedCompareStrings.get(objid))
            if len(trackedCompareStrings.get(objid)) >= 3:
                trackedDone[objid] = all_same(trackedCompareStrings.get(objid))
                if not trackedDone[objid] & len(trackedCompareStrings.get(objid)) > 3:
                    updated = trackedCompareStrings.get(objid)[1::]
                    trackedCompareStrings[objid] = updated
                if trackedDone[objid]:
                    # write to database
                    print('khalas')
                    with open('../recognition.csv', 'a', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        row = [outText, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                        writer.writerow(row)



os.chdir('videoresult')

# video code
while vid.isOpened():
    # function extract frames
    _, frame = vid.read()
    centerCurrent = []
    endpoints = []

    if frame is not None:
        count += 1

        frame = cv2.resize(frame, (480, 320))
        # run the YOLO model to detect license plates
        result = model(frame)

        # get the coordinates of the bounding boxes
        boxesCoordinates = result[0].boxes.xyxy

        for coord in boxesCoordinates:
            # use numpy not tensor
            box = coord.cpu().data.numpy()

            # get start and end points to draw box and crop the plate
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))

            centerCurrent.append((start_point, end_point))

            # draw a box on the detected plate
            frame = cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=2)

        tracker()



        if ocrFlag:
            # image postproc and OCR every 10 frames
            if count % 10 == 0:
                imageProc_ocr(frame)
            # writing the characters text
            doneCopy = trackedDone.copy()
            for objid, done in doneCopy.items():
                if not done:
                    continue
                if done:
                    ocr = trackedCompareStrings.get(objid)[0]
                    ocr = arabic_reshaper.reshape(ocr)
                    ocr = get_display(ocr)

                    fonttype = "arial.ttf"
                    font = ImageFont.truetype(fonttype, 32)
                    img_pil = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pil)
                    startPT = tracked.get(objid)[0]
                    bbox = draw.textbbox((startPT[0], startPT[1] - 30), ocr, font=font)
                    draw.rectangle(bbox, fill="black")
                    draw.text((startPT[0], startPT[1] - 30), ocr, font=font, fill="white")
                    frame = np.array(img_pil)

        centerPrev = centerCurrent.copy()
        # frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("fig", frame)

        cv2.waitKey(1)
