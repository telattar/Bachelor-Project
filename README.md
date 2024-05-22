# Egyptian License Plate Recognition in vehicles and motorcycles 🚗🛵
My bachelor project, YOLOv8n-based.

In this project, I developed a deep learning-based model that can detect and recognize the Egyptian license plates attached to cars and motorcycles. The model was trained on a dataset of different license plates with a variation of Arabic letters and Indian numerals.

### Tech/Framework Used
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
<img src="https://github.com/telattar/Bachelor-Project/assets/110330655/009b86b4-1e5e-432a-87e6-b7e86cf76e90" width="90">
<img src="https://github.com/telattar/Bachelor-Project/assets/110330655/79cd9ecf-07b2-4cbc-9762-95031c93a33b" width="40">

## Project
This project proposed a real-time deep learning-based standard Egyptian license plate recognition system using YOLOv8 and EasyOCR.

To the best of my knowledge, this is the first proposed system that is capable of recognizing Egyptian motorcycle license plates. A 1200-image dataset of annotated motorcycle license plates has been collected. This dataset is publicly available at [this repository](https://github.com/telattar/Egyptian-motorcycle-license-plate-dataset).

Several image post-processing procedures, such as contrast and skew adjustments, image sharpening, super resolution, and license plate cropping, were used to improve the system’s accuracy. The method saves the recognition result to a `.csv` file for further use. The detection step’s accuracy was determined to be 100%, while the recognition accuracy was 80%.

### Methodology

#### 1. Training the Model
The YOLO model was trained on both the [EALPR](https://github.com/ahmedramadan96/EALPR) dataset and a the previously mentioned motorcycle image dataset. Training iterations were conducted for 25, 50, 100, and 150 epochs, comparing the accuracy of each resulting model.

#### 2. License Plate Detection
To ensure the system's speed and accuracy for real-time use, YOLOv8 was chosen for license plate detection. This model can accurately identify the location of a license plate in approximately 1 millisecond per frame.

#### 3. License Plate Tracking
Each new vehicle entering the scene is assigned an identification number for its license plate, which is tracked until it leaves. Tracking is achieved by comparing the center points of the plate in consecutive frames, identifying the same object if the distance difference is minimal.

#### 4. Image Processing
1. **Cropping the Detected Plate:** The standard Egyptian license plate, with a 2:1 aspect ratio, is cropped to exclude the top portion containing the country’s name and background color, focusing on the character and numeric sections.
2. **Contrast Adjustment:** The image contrast is adjusted based on lighting conditions using standard deviation to ensure optimal OCR accuracy.
3. **Super Resolution:** PyPi’s super resolution module is used to enhance the resolution of the cropped license plate image to improve OCR accuracy.
4. **Image Sharpening:** Post super-resolution, the image is sharpened using PIL’s image sharpening function to enhance edge clarity for better OCR results.
5. **Skew Adjustment:** Canny Edge Detection and Hough Line Transform are used to detect and correct the skew of the license plate image to ensure proper character recognition.

#### 5. Character Recognition using [EasyOCR](https://github.com/JaidedAI/EasyOCR)
EasyOCR was chosen for its accuracy and efficiency, as it did not require writing images to disk.

#### 6. Character Anomaly Correction
Anomalous characters, which are not used in Egyptian license plates, were identified and manually corrected. This step ensures the accuracy of the OCR results by replacing incorrect characters with the correct ones.

<img src="https://github.com/telattar/Bachelor-Project/assets/110330655/f45c83f8-8633-4258-b300-5ffc2c59693c" width="400">


#### 7. Saving the Recognition Results
The recognized car number and the recognition time are saved in a comma-delimited file. This data is useful for security systems to track the time and location of vehicle identification.

## Installation
1. Please make sure there is a working webcam on your device. If there is not, feel free to use your mobile phone camera as an integrated camera using a third-party application like [Camo](https://reincubate.com/camo/).
2. Clone the project to your preferred directory. 
3. Open the unzipped folder in PyCharm IDE.
4. Open a new terminal and install all the dependencies by running the command `pip install -r requirements.txt`
5. Run the file `main.py`. You know that the project is properly running when your webcam turns on and a live video stream is displayed on the screen.
6. Put the **EGYPTIAN** license plate in front of the camera (you can use your mobile phone to show a photo to the cam) and the system should start by drawing a green box around the plate.
7. Wait for the OCR process to complete. Please be patient, it might need a few seconds to make sure that the recognized characters are correct.
8. Once the characters are plotted on the live video, that is when the process is complete.
9. Open the file recognition.csv and check for the plate characters and the saving time.

## Citation

Please cite the link to this repo if this project helps your research. A publication citation will be uploaded soon.
