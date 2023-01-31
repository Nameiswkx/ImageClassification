package com.example.imageclassification;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.TargetApi;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.TestLooperManager;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;



import com.example.imageclassification.ml.Model0;
import com.example.imageclassification.ml.Model1;
import com.example.imageclassification.ml.Model2;
import com.example.imageclassification.ml.Model3;
import com.example.imageclassification.ml.Model4;
import com.example.imageclassification.ml.Model5;
import com.example.imageclassification.ml.Model6;
import com.example.imageclassification.ml.Model7;
import com.example.imageclassification.ml.Model8;
import com.example.imageclassification.ml.Model9;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    TextView result,uncertainty,ood;
    Button gallery, camera;
    int imagewidthSize = 100;
    int imageheightSize = 75;
    ImageView imageView;


    int CAMERA_PICTURE = 3;

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == CAMERA_PICTURE && resultCode == RESULT_OK) {
            Bitmap picture = (Bitmap) data.getExtras().get("data");
            //because the CNN trained the pictures in squares, so resize the image.
            //change image size to dimension size, dimension is the min size from the image.
            int dimension = Math.min(picture.getWidth(), picture.getHeight());
            //recale the image and make it fittable size.
            picture = ThumbnailUtils.extractThumbnail(picture, picture.getWidth(), picture.getHeight());
            imageView.setImageBitmap(picture);
//            picture = Bitmap.createScaledBitmap(picture, imagewidthSize, imageheightSize, false);
            // convert to ARGB_8888
            picture = picture.copy(Bitmap.Config.ARGB_8888,true) ;
            //Using model to classify the picture.
            classifyImage(picture);
        } else {
            Uri dat = data.getData();
            Bitmap picture = null;
            try {
                picture = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
            } catch (IOException e) {
                e.printStackTrace();
            }
            imageView.setImageBitmap(picture);
            picture = Bitmap.createScaledBitmap(picture, imagewidthSize, imageheightSize, false);
            classifyImage(picture);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        uncertainty = findViewById(R.id.uncertainty);
        ood = findViewById(R.id.outofDistribution);

        camera.setOnClickListener(new View.OnClickListener() {


            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, CAMERA_PICTURE);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    //cancel annotation here!!!!
    public void classifyImage(Bitmap image) {
//        File model0_file = new File("/app/src/main/ml/model_0.tflite");

        try {
            Model0 model0 = Model0.newInstance(getApplicationContext());
            Model1 model1 = Model1.newInstance(getApplicationContext());
            Model2 model2 = Model2.newInstance(getApplicationContext());
            Model3 model3 = Model3.newInstance(getApplicationContext());
            Model4 model4 = Model4.newInstance(getApplicationContext());
            Model5 model5 = Model5.newInstance(getApplicationContext());
            Model6 model6 = Model6.newInstance(getApplicationContext());
            Model7 model7 = Model7.newInstance(getApplicationContext());
            Model8 model8 = Model8.newInstance(getApplicationContext());
            Model9 model9 = Model9.newInstance(getApplicationContext());

            System.out.println("loaded models");
            //normalization values
            float mean = 159.76f;
            float std = 46.44f;

            TensorImage tensorImage = TensorImage.fromBitmap(image);
            System.out.println("loaded image");
            ImageProcessor processor = new ImageProcessor.Builder()
                    .add(new ResizeOp(imageheightSize, imagewidthSize, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                    .add(new NormalizeOp(mean, std))
                    .build();
            TensorImage processedImage = processor.process(tensorImage);
            TensorBuffer inputBuffer = processedImage.getTensorBuffer();
            System.out.println("shape input buffer: " + Arrays.toString(inputBuffer.getShape()));

            // Runs model inference and gets result.
            float[][] softmax_outputs = new float[10][7];
            Model0.Outputs outputs0 = model0.process(inputBuffer);
            softmax_outputs[0] = outputs0.getOutputFeature0AsTensorBuffer().getFloatArray();
            Model1.Outputs outputs1 = model1.process(inputBuffer);
            softmax_outputs[1] = outputs1.getOutputFeature0AsTensorBuffer().getFloatArray();
            Model2.Outputs outputs2 = model2.process(inputBuffer);
            softmax_outputs[2] = outputs2.getOutputFeature0AsTensorBuffer().getFloatArray();
            Model3.Outputs outputs3 = model3.process(inputBuffer);
            softmax_outputs[3] = outputs3.getOutputFeature0AsTensorBuffer().getFloatArray();
            Model4.Outputs outputs4 = model4.process(inputBuffer);
            softmax_outputs[4] = outputs4.getOutputFeature0AsTensorBuffer().getFloatArray();
            Model5.Outputs outputs5 = model5.process(inputBuffer);
            softmax_outputs[5] = outputs5.getOutputFeature0AsTensorBuffer().getFloatArray();
            Model6.Outputs outputs6 = model6.process(inputBuffer);
            softmax_outputs[6] = outputs6.getOutputFeature0AsTensorBuffer().getFloatArray();
            Model7.Outputs outputs7 = model7.process(inputBuffer);
            softmax_outputs[7] = outputs7.getOutputFeature0AsTensorBuffer().getFloatArray();
            Model8.Outputs outputs8 = model8.process(inputBuffer);
            softmax_outputs[8] = outputs8.getOutputFeature0AsTensorBuffer().getFloatArray();
            Model9.Outputs outputs9 = model9.process(inputBuffer);
            softmax_outputs[9] = outputs9.getOutputFeature0AsTensorBuffer().getFloatArray();

            // Releases models resources if no longer used.
            model0.close();
            model1.close();
            model2.close();
            model3.close();
            model4.close();
            model5.close();
            model6.close();
            model7.close();
            model8.close();
            model9.close();


//            String[] classes = {"Melanocytic nevi", "Dermatofibroma", "Vascular lesions", "Melanoma", "Benign keratosis-like lesions", "Basal cell carcinoma", "Actinic keratoses"};
            String[] classes = {"Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions", "Dermatofibroma", "Melanocytic nevi", "Melanoma", "Vascular lesions"};


            // compute prediction and uncertainty using variation ratio
            // that means: plurality vote for prediction and for uncertainty: 1 - w / S,
            //    where w is the number of samples where the overall prediction equals the prediction of the sample
            //    and S is the total number of samples.

            HashMap<Integer, Integer> votes = new HashMap<>();
            float[] mean_distr = new float[7];
            for(int i = 0; i < 10; ++i){
                System.out.println("-----");
                float currMax = -1;
                int currMaxIdx = -1;
                for(int j = 0; j< 7; ++j) {
                    mean_distr[j] += softmax_outputs[i][j] / 7;
                    float output = softmax_outputs[i][j];
                    if (output > currMax) {
                        currMax = output;
                        currMaxIdx = j;
                    }
                }
                if(votes.containsKey(currMaxIdx)) {
                    votes.put(currMaxIdx, votes.get(currMaxIdx) + 1);
                }
                else {
                    votes.put(currMaxIdx, 1);
                }
            }

            int prediction = Collections.max(votes.entrySet(), (entry1, entry2) -> entry1.getValue() - entry2.getValue()).getKey();
            double uncertainty_val = 1.0 - (double) votes.get(prediction) / (double) softmax_outputs.length;
            String predicted_class = classes[prediction];

            result.setText(predicted_class);
            String rounded_uncert = Double.toString(Math.round(uncertainty_val * 100.0) / 100.0) + "\n";
            uncertainty.setText(rounded_uncert);

            ood.setText("");
            if ((Math.round(uncertainty_val * 100.0) / 100.0)> 0.2){
                ood.setText("It can be considered as Out-of-Distribution");
            }


            String s = "";
            for (int i =0; i<classes.length; i++){
                s+= classes[i] + mean_distr[i] + "\n";
            }
            System.out.println("Mean distribution: \n" + s);
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
}