package e.nd.paddatacapture;

import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.util.Base64;
import android.util.Base64OutputStream;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class UploadWorker extends Worker {
    public UploadWorker(@NonNull Context context, @NonNull WorkerParameters params) {
        super(context, params);
    }

    @Override
    public Result doWork() {
        // Build Request
        Uri.Builder builder = new Uri.Builder();
        builder.scheme("https")
                .authority("pad.crc.nd.edu")
                .appendPath("index.php")
                .appendQueryParameter("option", "com_jbackend")
                .appendQueryParameter("view", "request")
                .appendQueryParameter("module", "querytojson")
                .appendQueryParameter("resource", "upload")
                .appendQueryParameter("action", "post");

        String originalImage = null;
        String rectifiedImage = null;

        StringBuilder sbParams = new StringBuilder();
        try {
            sbParams.append("api_key").append("=").append(URLEncoder.encode("D5HDZG76N3ICA3GBUYWC", "UTF-8")).append("&");
            sbParams.append("category_name").append("=").append(URLEncoder.encode("JCSTest", "UTF-8")).append("&");
            sbParams.append("camera1").append("=").append(URLEncoder.encode(Build.MANUFACTURER + " " + Build.MODEL, "UTF-8")).append("&");

            String sampleName = getInputData().getString("SAMPLE_NAME");
            if (sampleName == null ){
                return Result.failure();
            }
            sbParams.append("sample_name").append("=").append(URLEncoder.encode(sampleName, "UTF-8")).append("&");

            String sampleId = getInputData().getString("SAMPLE_ID");
            if (sampleId == null ){
                return Result.failure();
            }
            sbParams.append("sampleid").append("=").append(URLEncoder.encode(sampleId, "UTF-8")).append("&");

            String notes = getInputData().getString("NOTES");
            if (notes == null ){
                return Result.failure();
            }
            sbParams.append("notes").append("=").append(URLEncoder.encode(notes, "UTF-8")).append("&");

            String quantity = getInputData().getString("QUANTITY");
            if (quantity == null ){
                return Result.failure();
            }
            sbParams.append("quantity").append("=").append(URLEncoder.encode(quantity, "UTF-8")).append("&");

            String timestamp = getInputData().getString("TIMESTAMP");
            if(timestamp == null) {
                return Result.failure();
            }
            sbParams.append("file_name").append("=").append(URLEncoder.encode("capture." + timestamp + ".png", "UTF-8")).append("&");
            sbParams.append("file_name2").append("=").append(URLEncoder.encode("rectified." + timestamp + ".png", "UTF-8")).append("&");

            originalImage = getInputData().getString("ORIGINAL_IMAGE");
            if(originalImage == null) {
                return Result.failure();
            }
            String origianlB64 = FileToBase64(Uri.parse(originalImage));
            sbParams.append("uploaded_file").append("=").append(URLEncoder.encode(origianlB64, "UTF-8")).append("&");
            sbParams.append("hash_file1").append("=").append(URLEncoder.encode(MD5(origianlB64), "UTF-8")).append("&");

            rectifiedImage = getInputData().getString("RECTIFIED_IMAGE");
            if(rectifiedImage == null) {
                return Result.failure();
            }
            String rectifiedB64 = FileToBase64(Uri.parse(rectifiedImage));
            sbParams.append("uploaded_file2").append("=").append(URLEncoder.encode(rectifiedB64, "UTF-8")).append("&");
            sbParams.append("hash_file2").append("=").append(URLEncoder.encode(MD5(rectifiedB64), "UTF-8"));
        }catch(Exception e) {
            e.printStackTrace();
        }

        // Send Data
        try{
            URL urlObj = new URL(builder.build().toString());

            HttpURLConnection conn = (HttpURLConnection) urlObj.openConnection();
            conn.setDoOutput(true);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Accept-Charset", "UTF-8");
            conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
            conn.setRequestProperty("charset", "utf-8");
            conn.setReadTimeout(10000);
            conn.setConnectTimeout(15000);
            conn.connect();

            String paramsString = sbParams.toString();

            DataOutputStream wr = new DataOutputStream(conn.getOutputStream());
            wr.writeBytes(paramsString);
            wr.flush();
            wr.close();

            Log.i("Test", String.format("%d", conn.getResponseCode()));

            InputStream in = new BufferedInputStream(conn.getInputStream());
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));
            StringBuilder result = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                result.append(line);
            }

            Log.d("Test", "result from server: " + result.toString());
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Cleanup
        if( originalImage != null ){
            File file = new File(new File(originalImage).getParent());
            file.delete();
        }

        // Indicate whether the work finished successfully with the Result
        return Result.success();
    }

    private String FileToBase64(Uri path) {
        byte[] bytes;
        byte[] buffer = new byte[8192];
        int bytesRead;
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try {
            InputStream inputStream = getApplicationContext().getContentResolver().openInputStream(path);
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                output.write(buffer, 0, bytesRead);
            }
        } catch (IOException e) {
            Log.e("UT", "base64encode: output.write failed", e);
            return null;
        }
        bytes = output.toByteArray();
        return Base64.encodeToString(bytes, Base64.NO_WRAP);
    }

    public String MD5(String s) {
        try {
            // Create MD5 Hash
            MessageDigest digest = java.security.MessageDigest.getInstance("MD5");
            digest.update(s.getBytes());
            byte messageDigest[] = digest.digest();

            // Create Hex String
            StringBuffer hexString = new StringBuffer();
            for (int i=0; i<messageDigest.length; i++) {
                hexString.append(String.format("%1$02X",(0xFF & messageDigest[i])));
            }

            return hexString.toString().toLowerCase();
        }catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return "";
    }
}
