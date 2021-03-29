package e.nd.paddatacapture;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.Toolbar;
import android.text.format.DateUtils;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;

import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.label.TensorLabel;
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

    SharedPreferences preferences;

    String qr = null;
    String timestamp = null;

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i("GBT", "onCreate");

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

        // setup remainder
        this.preferences = initializePreferences("Testing");
        setContentView(R.layout.activity_main);
        Toolbar myToolbar = (Toolbar) findViewById(R.id.my_toolbar);
        setSupportActionBar(myToolbar);
        try {
            SetUpInputs();
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    private void startImageCapture(){
        Log.i("GBR", "Image capture starting");
        if(this.qr != null) {
            Log.i("GBR", this.qr);
        }
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
        Log.i("GBT", "onResume");
        if(this.qr == null){
            Log.i("GBR", "Calling image capture from start");
            startImageCapture();
        } else {
            Log.i("GBR", "Tried to call image capture when I shouldn't have. ;)");
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
                    UncompressOutputs(getContentResolver().openInputStream(resultData), this.getCacheDir());

                    File rectifiedFile = new File(this.getCacheDir(), "rectified.png");

                    // Update UI
                    ImageView imageView = findViewById(R.id.imageView);
                    imageView.setImageURI(Uri.fromFile(rectifiedFile));

                    if (data.hasExtra("qr")) {
                        this.qr = data.getExtras().getString("qr");

                        TextView textView = findViewById(R.id.idText);
                        textView.setText(parseQR(this.qr));
                    }

                    if (data.hasExtra("timestamp")) {
                        this.timestamp = data.getExtras().getString("timestamp");
                        TextView textView = findViewById(R.id.timeText);

                        textView.setText(this.timestamp);
                    }

                    // crop input image
                    Bitmap bm = BitmapFactory.decodeFile(rectifiedFile.getPath());
                    bm = Bitmap.createBitmap(bm, 71, 359, 636, 490);

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
                            if (num_mod != number_of_models - 1) {
                                output_string += ", ";
                            }

                            // print results
                            Log.i("GBR", String.valueOf(probabilityBuffer[num_mod].getFloatArray()[0]));
                            Log.i("GBR", String.valueOf(probabilityBuffer[num_mod].getFloatArray()[maxidx]));
                            Log.i("GBR", associatedAxisLabels[num_mod].get(maxidx));
                        }
                    }

                    // TODO: This is a temporary output position
                    // Use % input for long term
                    TextView textView1 = (TextView)findViewById(R.id.batchAuto);
                    textView1.setTextSize(18);
                    textView1.setText("  " + output_string + "%");

                    Log.i("GBR", output_string + "%");
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        }
        else if (requestCode == 11) {
            Log.i("GBR", "Calling from email done");
            startImageCapture();
        }

        Log.i("GBR", String.valueOf(resultCode));
        Log.i("GBR", String.valueOf(requestCode));
    }

    private ArrayList<String> getAll(JSONObject obj) throws JSONException {
        String str2 = (String) obj.get("All");
        str2 = str2.replace("[", "");
        str2 = str2.replace("]", "");
        str2 = str2.replace(" ", "");
        ArrayList<String> myList = new ArrayList<String>(Arrays.asList(str2.split(",")));
        return myList;
    }

    private void SetUpInputs() throws JSONException {
        ArrayList<String> drugList;
        String target;
        if( associatedAxisLabels[0].size() > 0 ) {
            drugList = (ArrayList) associatedAxisLabels[0];
            target = associatedAxisLabels[0].get(0);
        } else {
            JSONObject obj = new JSONObject(this.preferences.getString("Drugs", ""));
            drugList = getAll(obj);
            target = obj.getString("Last");
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,
                android.R.layout.simple_spinner_dropdown_item, drugList);
        Spinner spinner = (Spinner)findViewById(R.id.drugSpinner);
        spinner.setAdapter(adapter);
        spinner.setSelection(adapter.getPosition(target));
        JSONObject obj2 = new JSONObject(this.preferences.getString("Brands", ""));
        ArrayList<String> brandList = getAll(obj2);
        String target2 = obj2.getString("Last");
        ArrayAdapter<String> adapter1 = new ArrayAdapter<String>(this,
                android.R.layout.simple_spinner_dropdown_item, brandList);
        Spinner spinner1 = (Spinner)findViewById(R.id.brandSpinner);
        spinner1.setAdapter(adapter1);
        spinner1.setSelection(adapter1.getPosition(target2));
    }

    private SharedPreferences initializePreferences(String project){
        SharedPreferences sharedPreferences = getSharedPreferences(project, MODE_PRIVATE);
        Boolean ready = sharedPreferences.getBoolean("Initialized", false);
        if(ready){
            try {
                JSONObject obj = new JSONObject(sharedPreferences.getString("Drugs", ""));
                ArrayList<String> myList = getAll(obj);
                Log.i("GBP", myList.toString());
            } catch (JSONException e) {
                e.printStackTrace();
            }
            return sharedPreferences;
        } else {
            SharedPreferences.Editor editor = sharedPreferences.edit();
            ArrayList<String> drugs;
            if( associatedAxisLabels[0].size() > 0 ) {
                drugs = (ArrayList)associatedAxisLabels[0];
                Log.i("GBP", "set labels");
            }else{
                drugs = Defaults.Drugs;
                Log.i("GBP", "Standard labels");
            }
            ArrayList<String> brands = Defaults.Brands;
            ArrayList<String> batches = Defaults.Batches;
            JSONObject obj = new JSONObject();
            try {
                obj.put("Last", "unknown");
                obj.put("All", drugs);
                editor.putString("Drugs", obj.toString());
                obj.put("Last", "100%");
                obj.put("All", brands);
                editor.putString("Brands", obj.toString());
                obj.put("Last", "n/a");
                obj.put("All", batches);
                editor.putString("Batches", obj.toString());
            } catch (JSONException e) {
                e.printStackTrace();
            }
            editor.putBoolean("Initialized", true);
            editor.commit();
            Log.i("GBP", "Initialized preferences for project");
            Log.i("GBP", project);
            return sharedPreferences;
        }
    }

    private void savePreference(String selected, String category) {
        JSONObject obj = null;
        try {
            obj = new JSONObject(this.preferences.getString(category, ""));
            ArrayList<String> opts = getAll(obj);
            Boolean present = opts.contains(selected);
            Log.i("GBP", present.toString());
            SharedPreferences.Editor editor = this.preferences.edit();
            if(false == present){
                opts.add(selected);
                obj.put("All", opts);
            }
            obj.put("Last", selected);
            editor.putString(category, obj.toString());
            editor.commit();
        } catch (JSONException e) {
            e.printStackTrace();
        }
        Log.i("GBP", obj.toString());
    }

    private String getDrug() {
        Spinner spinner = (Spinner)findViewById(R.id.drugSpinner);
        String ret = String.valueOf(spinner.getSelectedItem());
        savePreference(ret, "Drugs");
        return ret;
    }

    private String getBrand() {
        Spinner spinner = (Spinner)
                findViewById(R.id.brandSpinner);
        String ret = String.valueOf(spinner.getSelectedItem());
        if(ret.isEmpty()){
            ret = "100%";
        }
        savePreference(ret, "Brands");
        return ret;
    }

    private int getPercentage(String raw) {
        int ret = 100;
        String trimmed = raw.substring(0, raw.length() - 1);
        Log.i("GB", trimmed);
        try {
            ret = Integer.parseInt(trimmed);
        } catch (NumberFormatException nfe){
            Toast.makeText(this, nfe.toString(), Toast.LENGTH_LONG);
        }
        return ret;
    }

    private String getBatch() {
        TextView textView1 = (TextView)
                findViewById(R.id.batchAuto);
        String ret = String.valueOf(textView1.getText());
        if(ret.isEmpty()){
            ret = "n/a";
        }
        ret = ret.toLowerCase();
        savePreference(ret, "Batches");
        return ret;
    }

    private String getNotes() {
        EditText editText = (EditText) findViewById(R.id.editText);
        String ret = String.valueOf(editText.getText());
        return ret;
    }

    public void sendEmail(View view) {
        Log.i("GB", "Button pushed");

        Intent emailIntent = new Intent(Intent.ACTION_SEND_MULTIPLE);
        emailIntent.setType("message/rfc822");
        emailIntent.setType("application/image");
        String[] target = {"paper.analytical.devices@gmail.com"};
        Uri attachment = buildJSON();

        Log.i("GB", attachment.toString());
        ArrayList<Uri> attachments = new ArrayList<Uri>();
        attachments.add(attachment);
        attachments.add(FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", new File(this.getCacheDir(), "original.png")));
        attachments.add(FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", new File(this.getCacheDir(), "rectified.png")));

        emailIntent.putExtra(Intent.EXTRA_EMAIL, target);
        emailIntent.putExtra(Intent.EXTRA_SUBJECT, "PADs");
        emailIntent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        emailIntent.putParcelableArrayListExtra(Intent.EXTRA_STREAM, attachments);
        try {
            startActivityForResult(emailIntent, 11);
            Log.i("GBR", "Email client found");
        } catch (android.content.ActivityNotFoundException ex) {
            Log.i("GBR", "No email clients found");
        }
    }

    private Uri buildJSON() {
        Uri ret = Uri.EMPTY;
        try {
            File outputFile = File.createTempFile("data", ".json", this.getCacheDir());
            JSONObject jsonObject = new JSONObject();
            String compressedNotes = "Predicted drug =";
            compressedNotes += getBatch();
            compressedNotes += ", ";
            compressedNotes += getNotes();
            try {
                jsonObject.accumulate("sample_name", getDrug());
                jsonObject.accumulate("project_name", PROJECT);
                jsonObject.accumulate("camera1", Build.MANUFACTURER + " " + Build.MODEL);
                jsonObject.accumulate("sampleid", parseQR(this.qr));
                jsonObject.accumulate("qr_string", this.qr);
                jsonObject.accumulate("quantity", getPercentage(getBrand()));
                jsonObject.accumulate("notes", compressedNotes);
                jsonObject.accumulate("timestamp", this.timestamp);
            } catch (JSONException e) {
                e.printStackTrace();
            }
            FileWriter file = new FileWriter(outputFile);
            file.write(jsonObject.toString());
            file.flush();
            file.close();
            Log.i("GBR", outputFile.getPath());
            Log.i("GBR", getApplicationContext().getPackageName());
            ret = FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", outputFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    public void saveData(View view) {
        Log.i("GBR", "Pr-email");
        sendEmail(view);
    }

    public void discardData(View view) {
        try {
            SetUpInputs();
        } catch (JSONException e) {
            e.printStackTrace();
        }
        Log.i("GBR", "Calling image capture from discard");
        startImageCapture();
    }

    public String parseQR(String qr) {
        String outS = qr;
        if (qr.startsWith("padproject.nd.edu/?s=")){
            outS = qr.substring(21);
        } else if (qr.startsWith("padproject.nd.edu/?t=")){
            outS = qr.substring(21);
        }
        return outS;
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
