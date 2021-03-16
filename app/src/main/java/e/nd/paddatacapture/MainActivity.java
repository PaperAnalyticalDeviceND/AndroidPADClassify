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

import java.io.File;
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

public class MainActivity extends AppCompatActivity {
    static final String PROJECT = "FHI2020";
    SharedPreferences preferences;
    Uri raw = null;
    Uri rectified = null;
    String qr = null;
    String timestamp = null;
    File cDir = null;

    ImageProcessor imageProcessor = null;
    TensorImage tImage = null;
    TensorBuffer probabilityBuffer = null;
    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    // TODO: Declare a TFLite interpreter
    protected Interpreter tflite;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;
    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    final String ASSOCIATED_AXIS_LABELS = "labels.txt";
    List<String> associatedAxisLabels = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i("GBT", "onCreate");
        this.preferences = initializePreferences("Testing");
        setContentView(R.layout.activity_main);
        Toolbar myToolbar = (Toolbar) findViewById(R.id.my_toolbar);
        setSupportActionBar(myToolbar);
        try {
            SetUpInputs();
        } catch (JSONException e) {
            e.printStackTrace();
        }

        // Initialization code for TensorFlow Lite
        // Initialise the model
        try{
            tfliteModel
                    = FileUtil.loadMappedFile(this,
                    "model_small_1_10.tflite");

            // does it have metadata?
            MetadataExtractor metadata = new MetadataExtractor(tfliteModel);
            if(metadata.hasMetadata()) {
                // create new list
                associatedAxisLabels = new ArrayList<>();

                // get labels
                InputStream a = metadata.getAssociatedFile("labels.txt");
                BufferedReader r = new BufferedReader(new InputStreamReader(a));
                String line;
                while ((line = r.readLine()) != null) {
                    associatedAxisLabels.add(line);
                    //Log.e("GBR", line);
                }

                // other metadata
                ModelMetadata mm = metadata.getModelMetadata();
                Log.e("GBR", mm.description());

            }else{
                // load labels from file
                try {
                    associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
                } catch (IOException e) {
                    Log.e("GBR", "Error reading label file", e);
                }
            }

            // create interpreter
            tflite = new Interpreter(tfliteModel, tfliteOptions);

            // Reads type and shape of input and output tensors, respectively.
            // input
            int imageTensorIndex = 0;
            int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, 227, 227, 3}
            DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

            //output
            int probabilityTensorIndex = 0;
            // get output shape
            int[] probabilityShape =
                    tflite.getOutputTensor(0).shape(); // {1, NUM_CLASSES}
            Log.e("GBR", String.valueOf(probabilityShape[1]));
            DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

            // Create an ImageProcessor with all ops required. For more ops, please
            // refer to the ImageProcessor Architecture section in this README.
            imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(imageShape[2], imageShape[1], ResizeOp.ResizeMethod.BILINEAR))
                            .build();

            // Create a TensorImage object. This creates the tensor of the corresponding
            // tensor type DataType.FLOAT32.
            tImage = new TensorImage(imageDataType);

            // Create a container for the result and specify that this is not a quantized model.
            // Hence, the 'DataType' is defined as DataType.FLOAT32
            probabilityBuffer =
                    TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
                    //TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.FLOAT32);

        } catch (IOException e){
            Log.e("GBR", "Error reading model", e);
        }

    }

    private void startImageCapture(){
        Log.i("GBR", "Image capture starting");
        if(this.qr != null) {
            Log.i("GBR", this.qr);
        }
        if ((ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) | (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED)) {
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

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        Log.i("GBT", "onActivityResult");
        if (resultCode == RESULT_OK && requestCode == 10) {
            File rectifiedFile = null;
            Log.i("GBR", data.toString());
            if (data.hasExtra("raw")) {
                File file = new File(data.getExtras().getString("raw"));
                this.raw = FileProvider.getUriForFile(getApplicationContext(), getApplicationContext().getPackageName(), new File(file.getPath()));
                getApplicationContext().grantUriPermission(getApplicationContext().getPackageName(), this.raw, Intent.FLAG_GRANT_READ_URI_PERMISSION);
                //TODO: Fragile as hell, add error checks
                String path = file.getPath().replace("original.png", "");
                this.cDir = new File(path);
                Log.i("GBR", this.cDir.toString());
            } else {
                Log.i("GBR", "Raw missing");
            }
            if (data.hasExtra("rectified")){
                rectifiedFile = new File(data.getExtras().getString("rectified"));
                this.rectified = FileProvider.getUriForFile(getApplicationContext(), getApplicationContext().getPackageName(), new File(rectifiedFile.getPath()));
                getApplicationContext().grantUriPermission(getApplicationContext().getPackageName(), this.rectified, Intent.FLAG_GRANT_READ_URI_PERMISSION);
                Log.i("GBR", rectifiedFile.toString());
            } else {
                Log.i("GBR", "Rectified missing");
            }
            if (data.hasExtra("qr")){
                this.qr = data.getExtras().getString("qr");
                //Log.i("GBR", this.qr);
            } else {
                Log.i("GBR", "QR missing");
            }
            if (data.hasExtra("timestamp")){
                this.timestamp = data.getExtras().getString("timestamp");
                //Log.i("GBR", this.timestamp);
            } else {
                Log.i("GBR", "Timestamp missing");
            }
            updateExternalData();

            /*
             Analysis code for every frame
             Pre-process the image
            */

            //try {
            // crop input image
            Bitmap bm = BitmapFactory.decodeFile(rectifiedFile.getPath());
            bm = Bitmap.createBitmap(bm, 71, 359, 636, 490);
            //Log.i("GBR", String.valueOf(bm.getWidth()));

            // InputStream bitmap=getAssets().open("test_4.png");
            // Bitmap bit = BitmapFactory.decodeStream(bitmap);
            tImage.load(bm);
            tImage = imageProcessor.process(tImage);

            // Running inference
            if(null != tflite) {
                // categorize
                tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
                float[] probArray = probabilityBuffer.getFloatArray();
                int maxidx = findMaxIndex(probArray);

                // print results
                Log.i("GBR", String.valueOf(probabilityBuffer.getFloatArray()[3]));
                Log.i("GBR", String.valueOf(probabilityBuffer.getFloatArray()[maxidx]));
                Log.i("GBR", associatedAxisLabels.get(maxidx));

            }
//            } catch (IOException e1) {
//                // TODO Auto-generated catch block
//                e1.printStackTrace();
//            }

        }
        else if (requestCode == 11) {
            Log.i("GBR", "Calling from email done");
            startImageCapture();
        }

        //
        Log.i("GBR", String.valueOf(resultCode));
        Log.i("GBR", String.valueOf(requestCode));
    }

    private void updateExternalData(){
        if (this.rectified != null){
            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageURI(this.rectified);
        }
        if (this.timestamp != null){
            TextView textView = findViewById(R.id.timeText);
            textView.setText(this.timestamp);
        }
        if (this.qr != null){
            TextView textView = findViewById(R.id.idText);
            textView.setText(parseQR(this.qr));
        }
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
        JSONObject obj = new JSONObject(this.preferences.getString("Drugs", ""));
        ArrayList<String> drugList = getAll(obj);
        String target = obj.getString("Last");
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
        JSONObject obj3 = new JSONObject(this.preferences.getString("Batches", ""));
        ArrayList<String> batchList = getAll(obj3);
        String target3 = obj3.getString("Last");
        ArrayAdapter<String> adapter2 = new ArrayAdapter<String>(this,
                android.R.layout.simple_selectable_list_item, batchList);
        AutoCompleteTextView textView2 = (AutoCompleteTextView)
                findViewById(R.id.batchAuto);
        textView2.setAdapter(adapter2);
        textView2.setText(target3);
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
            ArrayList<String> drugs = testDrugs;
            ArrayList<String> brands = testBrands;
            ArrayList<String> batches = testBatches;
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
        AutoCompleteTextView textView1 = (AutoCompleteTextView)
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
        //String[] target = {"diyogon@gmail.com"};
        Uri attachment = buildJSON();

        Log.i("GB", attachment.toString());
        ArrayList<Uri> attachments = new ArrayList<Uri>();
        attachments.add(attachment);
        attachments.add(this.raw);
        attachments.add(this.rectified);

        emailIntent.putExtra(Intent.EXTRA_EMAIL, target);
        emailIntent.putExtra(Intent.EXTRA_SUBJECT, "PADs");
        emailIntent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        emailIntent.putParcelableArrayListExtra(Intent.EXTRA_STREAM, attachments);
        try {
            startActivityForResult(emailIntent, 11);
        } catch (android.content.ActivityNotFoundException ex) {
            Log.i("GBR", "No email clients found");
        }
    }

    private Uri buildJSON() {
        Uri ret = Uri.EMPTY;
        try {
            File outputFile = new File(this.cDir, "data.json");
            JSONObject jsonObject = new JSONObject();
            String compressedNotes = "batch=";
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
            ret = FileProvider.getUriForFile(getApplicationContext(), getApplicationContext().getPackageName(), new File(outputFile.getPath()));
            getApplicationContext().grantUriPermission(getApplicationContext().getPackageName(), ret, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    public void saveData(View view) {
        buildJSON();
        Log.i("GBR", "Calling image capture from save");
        startImageCapture();
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

    private static final ArrayList<String> testDrugs = new ArrayList<String>(){{
        add("unknown");
        add("albendazole");
        add("amoxicillin");
        add("ampicillin");
        add("ascorbic acid");
        add("azithromycin");
        add("benzyl penicillin");
        add("calcium carbonate");
        add("ceftriaxone");
        add("chloroquine");
        add("ciprofloxacin");
        add("doxycycline");
        add("epinephrine");
        add("ethambutol");
        add("ferrous sulfate");
        add("hydroxychloroquine");
        add("isoniazid");
        add("RI (rifampicin/isoniazid)");
        add("lactose");
        add("promethazine hydrochloride");
        add("pyrazinamide");
        add("rifampicin");
        add("RIPE");
        add("starch (maize)");
        add("sulfamethoxazole");
        add("talc");
        add("tetracycline");
    }};

    private static final ArrayList<String> testBrands = new ArrayList<String>(){{
        add("100%");
        add("80%");
        add("50%");
        add("20%");
    }};

    private static final ArrayList<String> testBatches = new ArrayList<String>(){{
        add("n/a");
    }};

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
