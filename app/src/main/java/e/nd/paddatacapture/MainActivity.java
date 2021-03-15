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

import java.nio.MappedByteBuffer;
import java.io.InputStream;
import java.util.List;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

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

        // Initialization code
        // Create an ImageProcessor with all ops required. For more ops, please
        // refer to the ImageProcessor Architecture section in this README.
        imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(227, 227, ResizeOp.ResizeMethod.BILINEAR))
                        .build();

        // Create a TensorImage object. This creates the tensor of the corresponding
        // tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
        tImage = new TensorImage(DataType.FLOAT32);

        // Create a container for the result and specify that this is a quantized model.
        // Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
        probabilityBuffer =
                TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.FLOAT32);

        // Initialise the model
        try{
            tfliteModel
                    = FileUtil.loadMappedFile(this,
                    "model_small_1_10.tflite");
            tflite = new Interpreter(tfliteModel, tfliteOptions);

            int[] probabilityShape =
                    tflite.getOutputTensor(0).shape();
            Log.e("GBR", String.valueOf(probabilityShape[0]));
            Log.e("GBR", String.valueOf(probabilityShape[1]));
        } catch (IOException e){
            Log.e("GBR", "Error reading model", e);
        }

        // load labels
        try {
            associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
        } catch (IOException e) {
            Log.e("GBR", "Error reading label file", e);
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
                File file = new File(data.getExtras().getString("rectified"));
                this.rectified = FileProvider.getUriForFile(getApplicationContext(), getApplicationContext().getPackageName(), new File(file.getPath()));
                getApplicationContext().grantUriPermission(getApplicationContext().getPackageName(), this.rectified, Intent.FLAG_GRANT_READ_URI_PERMISSION);
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
        }
        else if (requestCode == 11) {
            Log.i("GBR", "Calling from email done");
            startImageCapture();
        }

        //
        // Analysis code for every frame
        // Preprocess the image
        //Bitmap bitmap = BitmapFactory.decodeFile("test_4.png");
        Log.i("GBR","Pre-image");
        try {
            InputStream bitmap=getAssets().open("test_4.png");
            Bitmap bit = BitmapFactory.decodeStream(bitmap);
            tImage.load(bit);
            tImage = imageProcessor.process(tImage);
            Log.i("GBR","Post-image");
            // Running inference
            if(null != tflite) {
                Log.i("GBR","Pre-predict");
                tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
                Log.i("GBR","Post-predict");
                Log.i("GBR", String.valueOf(probabilityBuffer.getFloatArray()[3]));
                Log.i("GBR", String.valueOf(probabilityBuffer.getFloatArray()[4]));

                Log.i("GBR", associatedAxisLabels.get(4));

            }

        } catch (IOException e1) {
            // TODO Auto-generated catch block
            e1.printStackTrace();
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

}
