# AndroidPADClassify
## PAD_Classify
Android App based on Galen's PADDataCapture app which calls the AndroidPADCaptureBasic app (https://github.com/PaperAnalyticalDeviceND/AndroidPADCaptureBasic) via intents.

Origionally loaded a 227x227 png and classifies it using the ```test_small_1_10``` model, now loads the ```rectified.png``` image returned from AndroidPADCaptureBasic after cropping.

### 07/11/21. Pad_Classify now implements FHI360's PLSR method. 

The PLSR drug concentration results are printed after the NN concentrations as ```(PLS xx%)```.

For this aspect of the project a CSV is extracted from the PADS database and proccessed using Galan's ColorSelector project (https://github.com/PaperAnalyticalDeviceND/ColorSelector). The results are loaded into FHI360's R-shiny application (https://github.com/sortijas/PAD) to generate the coefficents per drug, ```PLS_COEFS_ALBENDAZOLE_10_REG_RGB_FULL_ALBEND_FIRST``` etc. with format,
```
	yy.36.comps
(Intercept)	35.7313572
A1.R	0.025729985
A1.G	-0.032458559
A1.B	0.012153512
A2.R	0.019541369
A2.G	-0.048318886
A2.B	0.067176884
...
```

These coefficents are combined using ```pls_generate_app_coeff_csv.py``` (from ColorSelector) and are stored in the file ```pls_coefficients.csv``` that is loaded into this app. with format,

```
albendazole	35.7313572	0.025729985	-0.032458559	0.012153512	...
```

There is a regression test for the Android code which is commented out in the initialization of ```Partial_least_squares```,

```
// run test at initialize
// calculate the concentration
// for regression should be 28.22623085
Bitmap bmp = BitmapFactory.decodeStream(context.getAssets().open("pls_test_16895.png"));
double concentration = do_pls(bmp, "albendazole");
```






