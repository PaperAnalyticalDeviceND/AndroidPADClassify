package e.nd.paddatacapture;

import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;
import androidx.work.Constraints;
import androidx.work.Data;
import androidx.work.NetworkType;
import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkManager;
import androidx.work.WorkRequest;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
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
import java.util.ArrayList;

public class ResultActivity extends AppCompatActivity {
    SharedPreferences mPreferences = null;

    String qr = "";
    String timestamp = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        // Setup Preferences
        mPreferences = getSharedPreferences(MainActivity.PROJECT, MODE_PRIVATE);

        // Setup compatability toolbar
        Toolbar myToolbar = findViewById(R.id.my_toolbar);
        setSupportActionBar(myToolbar);

        // Handle calling intent
        Intent intent = getIntent();

        ImageView imageView = findViewById(R.id.imageView);
        imageView.setImageURI(intent.getData());

        TextView vPredicted = findViewById(R.id.batchAuto);
        vPredicted.setText(intent.getStringExtra(MainActivity.EXTRA_PREDICTED));

        if( intent.hasExtra(MainActivity.EXTRA_SAMPLEID) ) {
            this.qr = intent.getStringExtra(MainActivity.EXTRA_SAMPLEID);
            TextView vSample = findViewById(R.id.idText);
            vSample.setText(parseQR(this.qr));
        }

        if( intent.hasExtra(MainActivity.EXTRA_TIMESTAMP) ) {
            this.timestamp = intent.getStringExtra(MainActivity.EXTRA_TIMESTAMP);
            TextView vTimestamp = findViewById(R.id.timeText);
            vTimestamp.setText(this.timestamp);
        }

        // Handle Drug List
        String tDrugs = "";
        ArrayAdapter<String> aDrugs = null;
        if( intent.hasExtra(MainActivity.EXTRA_LABEL_DRUGS) && intent.getStringArrayExtra(MainActivity.EXTRA_LABEL_DRUGS) != null ) {
            String[] drugs = intent.getStringArrayExtra(MainActivity.EXTRA_LABEL_DRUGS);
            aDrugs = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_dropdown_item, drugs);
            tDrugs = drugs[0];
        }else {
            aDrugs = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_dropdown_item, Defaults.Drugs);
            tDrugs = Defaults.Drugs.get(0);
        }

        Spinner sDrugs = findViewById(R.id.drugSpinner);
        sDrugs.setAdapter(aDrugs);
        sDrugs.setSelection(aDrugs.getPosition(mPreferences.getString("Drug", tDrugs)));

        // Handle Brands
        Spinner sBrands = findViewById(R.id.brandSpinner);
        ArrayAdapter<String> aBrands = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_dropdown_item, Defaults.Brands);
        sBrands.setAdapter(aBrands);
        sBrands.setSelection(aBrands.getPosition(mPreferences.getString("Brand", Defaults.Brands.get(0))));
    }

    @Override
    protected void onResume() {
        super.onResume();
        MainActivity.HoldCamera = false;
    }

    public void saveData(View view) {
        Log.i("GB", "Button pushed");

        String compressedNotes = "Predicted drug =";
        compressedNotes += getBatch();
        compressedNotes += ", ";
        compressedNotes += getNotes();

        Constraints constraints = new Constraints.Builder()
                .setRequiredNetworkType(NetworkType.UNMETERED)
                .build();

        WorkRequest myUploadWork =  new OneTimeWorkRequest.Builder(UploadWorker.class)
                .setConstraints(constraints)
                .addTag("result_upload")
                .setInputData(
                    new Data.Builder()
                            .putString("SAMPLE_NAME", getDrug())
                            .putString("SAMPLE_ID", parseQR(this.qr))
                            .putString("NOTES", compressedNotes)
                            .putString("QUANTITY", getPercentage(getBrand()))
                            .putString("TIMESTAMP", this.timestamp)
                            .putString("ORIGINAL_IMAGE", FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", new File(new File(this.getFilesDir(), timestamp), "original.png")).toString())
                            .putString("RECTIFIED_IMAGE", FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", new File(new File(this.getFilesDir(), timestamp), "rectified.png")).toString())
                        .build()
                )
                .build();

        WorkManager.getInstance(this).enqueue(myUploadWork);

        finish();
        /*
        Intent emailIntent = new Intent(Intent.ACTION_SEND_MULTIPLE);
        emailIntent.setType("message/rfc822");
        emailIntent.setType("application/image");
        String[] target = {"paper.analytical.devices@gmail.com"};
        Uri attachment = buildJSON();

        Log.i("GB", attachment.toString());
        ArrayList<Uri> attachments = new ArrayList<Uri>();
        attachments.add(attachment);
        attachments.add(FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", new File(new File(this.getFilesDir(), timestamp), "original.png")));
        attachments.add(FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", new File(new File(this.getFilesDir(), timestamp), "rectified.png")));

        emailIntent.putExtra(Intent.EXTRA_EMAIL, target);
        emailIntent.putExtra(Intent.EXTRA_SUBJECT, "PADs");
        emailIntent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        emailIntent.putParcelableArrayListExtra(Intent.EXTRA_STREAM, attachments);
        try {
            startActivityForResult(emailIntent, 11);
            Log.i("GBR", "Email client found");
        } catch (android.content.ActivityNotFoundException ex) {
            Log.i("GBR", "No email clients found");
        }*/
    }

    public void discardData(View view) {
        finish();
    }

    private Uri buildJSON() {
        Uri ret = Uri.EMPTY;
        try {
            JSONObject jsonObject = new JSONObject();
            String compressedNotes = "Predicted drug =";
            compressedNotes += getBatch();
            compressedNotes += ", ";
            compressedNotes += getNotes();
            try {
                jsonObject.accumulate("sample_name", getDrug());
                jsonObject.accumulate("project_name", MainActivity.PROJECT);
                jsonObject.accumulate("camera1", Build.MANUFACTURER + " " + Build.MODEL);
                jsonObject.accumulate("sampleid", parseQR(this.qr));
                jsonObject.accumulate("qr_string", this.qr);
                jsonObject.accumulate("quantity", getPercentage(getBrand()));
                jsonObject.accumulate("notes", compressedNotes);
                jsonObject.accumulate("timestamp", this.timestamp);
            } catch (JSONException e) {
                e.printStackTrace();
            }

            File outputFile = File.createTempFile("data", ".json", this.getCacheDir());

            FileWriter file = new FileWriter(outputFile);
            file.write(jsonObject.toString());
            file.flush();
            file.close();

            ret = FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", outputFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    private String getDrug() {
        Spinner spinner = findViewById(R.id.drugSpinner);
        String ret = String.valueOf(spinner.getSelectedItem());
        mPreferences.edit().putString("Drug", ret).commit();
        return ret;
    }

    private String getBrand() {
        Spinner spinner = findViewById(R.id.brandSpinner);
        String ret = String.valueOf(spinner.getSelectedItem());
        if(ret.isEmpty()){
            return "100%";
        }
        mPreferences.edit().putString("Brand", ret).commit();
        return ret;
    }

    private String getPercentage(String raw) {
        return raw.substring(0, raw.length() - 1);
    }

    private String getBatch() {
        TextView textView1 = findViewById(R.id.batchAuto);
        String ret = String.valueOf(textView1.getText());
        if(ret.isEmpty()){
            ret = "n/a";
        }
        return ret.toLowerCase();
    }

    private String getNotes() {
        EditText editText = findViewById(R.id.editText);
        return String.valueOf(editText.getText());
    }

    public String parseQR(String qr) {
        if (qr.startsWith("padproject.nd.edu/?s=") || qr.startsWith("padproject.nd.edu/?t=") ){
            return qr.substring(21);
        }
        return qr;
    }
}