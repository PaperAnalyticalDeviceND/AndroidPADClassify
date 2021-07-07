    package e.nd.paddatacapture;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.os.Bundle;
import android.util.Log;
import android.view.View;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import org.opencv.android.OpenCVLoader;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.metadata.schema.ModelMetadata;

import java.nio.MappedByteBuffer;
import java.io.InputStream;
import java.util.List;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class MainActivity extends AppCompatActivity {
    static final String PROJECT = "FHI360-App";
    public static boolean HoldCamera = false;

    public static final String EXTRA_SAMPLEID = "e.nd.paddatacapture.EXTRA_SAMPLEID";
    public static final String EXTRA_TIMESTAMP = "e.nd.paddatacapture.EXTRA_TIMESTAMP";
    public static final String EXTRA_PREDICTED = "e.nd.paddatacapture.EXTRA_PREDICTED";
    public static final String EXTRA_LABEL_DRUGS = "e.nd.paddatacapture.EXTRA_LABEL_DRUGS";

    // NN storage, now setting up array for multiple NN
    final int number_of_models = 2;
    String[] model_list = {"fhi360_small_1_21.tflite", "fhi360_conc_large_1_21.tflite"};

    ImageProcessor[] imageProcessor = {null, null};
    TensorImage[] tImage =  {null, null};
    TensorBuffer[] probabilityBuffer = {null, null};

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter[] tflite = {null, null};

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer[] tfliteModel = {null, null};

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options[] tfliteOptions = {new Interpreter.Options(), new Interpreter.Options()};

    final String[] ASSOCIATED_AXIS_LABELS = {"labels.txt", "labels.txt"};
    List<String>[] associatedAxisLabels = (ArrayList<String>[])new ArrayList[2];

    // pls class
    Partial_least_squares pls = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //initialize opencv
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
            Log.i("GBT", "Opencv not loaded");
        }

        // Initialization code for TensorFlow Lite
        // Initialise the models
        for(int num_mod=0; num_mod < number_of_models; num_mod++) {
            //final int num_mod = 0;
            try {
                tfliteModel[num_mod] = FileUtil.loadMappedFile(this, model_list[num_mod]);

                // does it have metadata?
                MetadataExtractor metadata = new MetadataExtractor(tfliteModel[num_mod]);
                if (metadata.hasMetadata()) {
                    // create new list
                    associatedAxisLabels[num_mod] = new ArrayList<>();

                    // get labels
                    InputStream a = metadata.getAssociatedFile("labels.txt");
                    BufferedReader r = new BufferedReader(new InputStreamReader(a));
                    String line;
                    while ((line = r.readLine()) != null) {
                        associatedAxisLabels[num_mod].add(line);
                    }

                    // other metadata
                    ModelMetadata mm = metadata.getModelMetadata();
                    Log.e("GBR", mm.description());
                    Log.e("GBR", mm.version());

                } else {
                    try {
                        associatedAxisLabels[num_mod] = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS[num_mod]);
                    } catch (IOException e) {
                        Log.e("GBR", "Error reading label file", e);
                    }
                }

                // create interpreter
                tflite[num_mod] = new Interpreter(tfliteModel[num_mod], tfliteOptions[num_mod]);

                // Reads type and shape of input and output tensors, respectively.
                int imageTensorIndex = 0;
                int[] imageShape = tflite[num_mod].getInputTensor(imageTensorIndex).shape(); // {1, 227, 227, 3}
                DataType imageDataType = tflite[num_mod].getInputTensor(imageTensorIndex).dataType();

                //output
                int probabilityTensorIndex = 0;

                // get output shape
                int[] probabilityShape =  tflite[num_mod].getOutputTensor(0).shape(); // {1, NUM_CLASSES}
                DataType probabilityDataType = tflite[num_mod].getOutputTensor(probabilityTensorIndex).dataType();

                // Create an ImageProcessor with all ops required. For more ops, please
                // refer to the ImageProcessor Architecture section in this README.
                imageProcessor[num_mod] = new ImageProcessor.Builder()
                                .add(new ResizeOp(imageShape[2], imageShape[1], ResizeOp.ResizeMethod.BILINEAR))
                                .build();

                // Create a TensorImage object. This creates the tensor of the corresponding
                // tensor type DataType.FLOAT32.
                tImage[num_mod] = new TensorImage(imageDataType);

                // Create a container for the result and specify that this is not a quantized model.
                // Hence, the 'DataType' is defined as DataType.FLOAT32
                probabilityBuffer[num_mod] = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
                //TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.FLOAT32);

            } catch (IOException e) {
                Log.e("GBR", "Error reading model", e);
            }
        }

        // setup pls
        pls = new Partial_least_squares(this);

        // setup remainder
        setContentView(R.layout.activity_main);
    }

    public void startImageCapture(View view){
        Log.i("GBR", "Image capture starting");
        if ((ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)  != PackageManager.PERMISSION_GRANTED)
                | (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED)) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 90);
        } else {
            Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse("pads://capture"));
            startActivityForResult(intent, 10);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if( !HoldCamera ) {
            startImageCapture(null);
        }
    }

    private void UncompressOutputs( InputStream fin, File targetDirectory ) throws Exception {
        byte[] buffer = new byte[4096];
        try (BufferedInputStream bis = new BufferedInputStream(fin); ZipInputStream stream = new ZipInputStream(bis)) {
            ZipEntry entry;
            while ((entry = stream.getNextEntry()) != null) {
                try (FileOutputStream fos = new FileOutputStream(targetDirectory.getPath() + "/" + entry.getName());
                     BufferedOutputStream bos = new BufferedOutputStream(fos, buffer.length)) {

                    int len;
                    while ((len = stream.read(buffer)) > 0) {
                        bos.write(buffer, 0, len);
                    }
                }
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        Log.i("GBT", "onActivityResult");
        if (resultCode == RESULT_OK && requestCode == 10) {
            Uri resultData = data.getData();
            if (resultData != null) {
                try {
                    // Create Output Directory
                    String timestamp = "";
                    if (data.hasExtra("timestamp")) timestamp = String.format("%d", data.getExtras().getLong("timestamp"));

                    File targetDir = new File(this.getFilesDir(), timestamp);
                    targetDir.mkdirs();

                    // Extract Files
                    UncompressOutputs(getContentResolver().openInputStream(resultData), targetDir);

                    // Handle rectified image
                    File rectifiedFile = new File(targetDir, "rectified.png");

                    // crop input image
                    Bitmap bmRect = BitmapFactory.decodeFile(rectifiedFile.getPath());
                    Bitmap bm = Bitmap.createBitmap(bmRect, 71, 359, 636, 490);

                    // create output string
                    String output_string = new String();

                    // categorize for each model in list
                    for (int num_mod = 0; num_mod < number_of_models; num_mod++) {
                        tImage[num_mod].load(bm);
                        tImage[num_mod] = imageProcessor[num_mod].process(tImage[num_mod]);

                        // Running inference
                        if (null != tflite[num_mod]) {
                            // categorize
                            tflite[num_mod].run(tImage[num_mod].getBuffer(), probabilityBuffer[num_mod].getBuffer());
                            float[] probArray = probabilityBuffer[num_mod].getFloatArray();
                            int maxidx = findMaxIndex(probArray);

                            // concat to output string
                            output_string += associatedAxisLabels[num_mod].get(maxidx);
                            if( num_mod == 0 ){
                                output_string += String.format(" (%.3f)", probabilityBuffer[num_mod].getFloatArray()[maxidx]);
                            }

                            if (num_mod != number_of_models - 1) {
                                output_string += ", ";
                            }
                            // print results
                            Log.i("GBR", String.valueOf(probabilityBuffer[num_mod].getFloatArray()[0]));
                            Log.i("GBR", String.valueOf(probabilityBuffer[num_mod].getFloatArray()[maxidx]));
                            Log.i("GBR", associatedAxisLabels[num_mod].get(maxidx));
                        }
                    }

                    // calculate concentration from PLSR method
                    double concentration = pls.do_pls(bmRect, "albendazole");

                    output_string += "%, (PLS " + (int)concentration + "%)";

                    Intent intent = new Intent(this, ResultActivity.class);
                    intent.setData(Uri.fromFile(rectifiedFile));
                    intent.putExtra(EXTRA_PREDICTED, output_string);
                    if (data.hasExtra("qr"))  intent.putExtra(EXTRA_SAMPLEID, data.getExtras().getString("qr"));
                    if (data.hasExtra("timestamp")) intent.putExtra(EXTRA_TIMESTAMP, timestamp);
                    if( associatedAxisLabels[0].size() > 0 ) intent.putExtra(EXTRA_LABEL_DRUGS, (String[]) associatedAxisLabels[0].toArray(new String[0]));
                    startActivity(intent);

                    HoldCamera = true;

                    Log.i("GBR", output_string + "%");
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        }

        Log.i("GBR", String.valueOf(resultCode));
        Log.i("GBR", String.valueOf(requestCode));
    }

    private static final int findMaxIndex(float [] arr) {
        float max = arr[0];
        int maxIdx = 0;
        for(int i = 1; i < arr.length; i++) {
            if(arr[i] > max) {
                max = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    };

}
