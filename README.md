# AndroidPADClassify
## PAD_Classify
Android App based on Galen's PADDataCapture app which calls the AndroidPADCaptureBasic app (https://github.com/PaperAnalyticalDeviceND/AndroidPADCaptureBasic) via intents.

Origionally loaded a 227x227 png and classifies it using the ```test_small_1_10``` model, now loads the ```rectified.png``` image returned from AndroidPADCaptureBasic after cropping.

