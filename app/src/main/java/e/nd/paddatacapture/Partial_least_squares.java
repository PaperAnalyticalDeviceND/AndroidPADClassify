package e.nd.paddatacapture;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import com.opencsv.CSVReader;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import static java.lang.String.valueOf;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class Partial_least_squares {
    // labels
    private List<String> labels = new ArrayList<String>();

    // coefficients
    private List<List<Float>> coeffs = new ArrayList<List<Float>>();

    // sizes
    int label_length, coeff_length = 0;

    // initialize
    public Partial_least_squares(Context context) {
        Log.i("GBT-PLS", "PLS created-");

        // load csv from assets
        try {
            // read csv file
            InputStream is = context.getAssets().open("pls_coefficients.csv");
            InputStreamReader reader = new InputStreamReader(is, Charset.forName("UTF-8"));
            List<String[]> csv = new CSVReader(reader).readAll();
            Log.i("GBT-PLS", csv.get(0)[0]);

            // get sizes
            final int label_length = csv.size();
            final int coeff_length = csv.get(0).length;

            // loop to put coeffs into matrix (List List)
            for(int i=0; i<label_length; i++){
                // load labels
                labels.add(csv.get(i)[0]);

                //add new coeff list
                coeffs.add(new ArrayList<Float>());

                // coeffs
                for(int j=1; j<coeff_length; j++){
                    coeffs.get(i).add(Float.parseFloat(csv.get(i)[j]));
                }
            }

            Log.i("GBT-PLS", csv.get(0)[0]);
            Log.i("GBT-PLS", valueOf(coeffs.get(0).get(0)) + ", " + labels.get(0));

            // run test at initialize
            // calculate the concentration
            // for regression should be 28.22623085
            Bitmap bmp = BitmapFactory.decodeStream(context.getAssets().open("pls_test_16895.png"));

            double concentration = do_pls(bmp, "albendazole");

        } catch (IOException e) {
            Log.i("GBT-PLS", "PLS error");
            e.printStackTrace();
        }

    }

    // actually perform the pls
    public double do_pls(Bitmap bmp, String drug){

        // find drug index
        int drug_idnx = labels.indexOf(drug);
        Log.i("GBT-PLS", "drug, "+drug+", index "+ drug_idnx);

        if (drug_idnx < 0){
            drug_idnx = 0;
        }

        // save results
        double concentration = 0.0;

        List<double []> results = new ArrayList<>();

        // load test image from drawable
        //try {
        //Mat BGRMat = Utils.loadResource(context, R.drawable.pls_test_16895);
        //Bitmap bmp = BitmapFactory.decodeStream(context.getAssets().open("pls_test_16895.png"));
        Mat BGRMat = new Mat();
        Utils.bitmapToMat(bmp, BGRMat);
        Imgproc.cvtColor(BGRMat,BGRMat,Imgproc.COLOR_RGBA2BGR);

        // CV_8U C3 is type 16
        Log.i("GBT-PLS", "Image load sizes, " + BGRMat.rows() +"," + BGRMat.cols() +"," + BGRMat.type());

        // hsv for intensity calcs
        Mat imgHSV = new Mat();
        cvtColor(BGRMat, imgHSV, Imgproc.COLOR_BGR2HSV);

        // get rectangles to process
        for (int lane=1; lane < 13; lane++){
            int laneStart = 17 + (53*lane) + 12;
            int laneEnd = 17 + (53*(lane+1)) - 12;
            //Log.i("GBT-PLS-lane", String.valueOf(lane));
            // for each region
            for (int region=0; region < 10; region++){
                //Log.i("GBT-PLS-region", String.valueOf(region));
                int start = 359;
                int totalLength = 273;
                int regionStart = start + (int)Math.floor((totalLength * region)/10);
                int regionEnd = start + (int)Math.floor((totalLength * (region+1))/10);

                Rect rroi = new Rect(laneStart, regionStart, laneEnd - laneStart, regionEnd - regionStart );
                //Log.i("GBT-PLS", String.valueOf(rroi) +","+regionStart+","+regionEnd+","+laneStart+","+laneEnd);

                Mat roi =  new Mat(imgHSV, rroi);
                Mat rgbROI =  new Mat(BGRMat, rroi);
                //Log.i("GBT-PLS", String.valueOf(roi.size()));

                // gt most intense pixels
                Vector<Point> pixels = findMaxIntensitiesFiltered(roi);
                //Log.i("GBT-PLS", "pix, "+String.valueOf(pixels.size()) +','+String.valueOf(pixels.get(0)));

                // average over rgb
                double [] rgb = avgPixels(pixels, rgbROI);

                // append
                results.add(rgb);

            }
        }

        // if we are here we should have results
        // get concentration
        concentration = calc_concentration(results, drug_idnx);

        // flag conc.
        Log.i("GBT-PLS", "PLS concentration, "+concentration);

        // return calculated concentration
        return concentration;

    }

    // Takes a HSV image and returns a list of the most intense pixels in it,
    // after applying filtering to minimize black bars on the edges
    private Vector<Point> findMaxIntensitiesFiltered(Mat img){
        Vector<Point> maxSet = new Vector<>();

        // imgS = img[:,:,1]
        // imgV = img[:,:,2]

        double maxI = 0;
        double centerX = (double)img.rows()/2.0;
        double centerY = (double)img.cols()/2.0;
        //Log.i("GBT-PLS", String.valueOf(centerX) + ','+String.valueOf(centerY));

        // loop over x
        for (int i=0; i<img.rows(); i++) {
            double dX = Math.abs(centerX - i);

            // loop over y
            for (int j=0; j<img.cols(); j++) {
                double dY = Math.abs(centerY - j);
                double sF = cosCorrectFactor(dX, dY, centerX, centerY);
                //Log.i("GBT-PLS", i +","+j);
                double cS = sF * img.get(i, j)[1]; //imgS
                double cV = sF * img.get(i, j)[2]; //imgV

                // black line?
                if (cS <= 35 && cV <= 70) {
                    //continue;
                }else if (cS > maxI) {
                    maxI = (int)img.get(i, j)[1]; //imgS
                    Point point = new Point(i, j);
                    maxSet.clear();
                    maxSet.add(point);
                }else if (cS == maxI) {
                    Point point = new Point(i, j);
                    maxSet.add(point);
                }
            }
        }

        return maxSet;
    }

    // Takes a distance from a center and returns a weight between 0 and 1
    // determined by cosine such that a point at the center has weight 1,
    // and a point at the extremes has weight ~0.
    public double cosCorrectFactor(double dx, double dy, double centerX, double centerY) {
        double relevantD = Math.max(((double)dx / (double)centerX), (double)dy / (double)centerY);
        double relevnatDRads = (Math.PI / 2.0) * relevantD;
        return Math.cos(relevnatDRads);
    }

    // Takes a list of pixels and a BGR image and returns the average
    // RGB pixel values
    private double [] avgPixels(Vector<Point> pixels, Mat img) {
        double totalB = 0;
        double totalG = 0;
        double totalR = 0;

        //  accumulate pixels
        for (int pixeli=0; pixeli < pixels.size(); pixeli++){
            int x = (int)pixels.get(pixeli).x;
            int y = (int)pixels.get(pixeli).y;
            double [] bgr = img.get(x, y);
            totalB += bgr[0];
            totalG += bgr[1];
            totalR += bgr[2];
        }

        // normalize if values exist
        if (pixels.size() != 0) {
            totalB /= pixels.size();
            totalG /= pixels.size();
            totalR /= pixels.size();
        }

        // return averages
        double[] ret = new double[]{totalR, totalG, totalB};
        return ret;
    }

    // calculate concentration using PLSR
    private double calc_concentration(List<double []> results, int drug_idnx){
        double conc = 0.0;

        // get coefficients to use
        List<Float> drug_coeffs = coeffs.get(drug_idnx);

        // first coeff is intercept
        conc = drug_coeffs.get(0);

        // do the pls coeff multiply
        for(int i=0; i<results.size(); i++){
            for(int j=0; j<3; j++) {
                conc += results.get(i)[j] * drug_coeffs.get(i * 3 + j + 1);
            }
        }

        //Log.i("GBT-PLS", "res size, "+String.valueOf(results.size())+","+drug_coeffs.size()+","+conc);
        // return concentration
        return conc;
    }
}