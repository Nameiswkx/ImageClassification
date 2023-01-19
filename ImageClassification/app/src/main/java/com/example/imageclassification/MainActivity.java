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



import com.example.imageclassification.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result,confidence;
    Button gallery, camera;
    int imagewidthSize = 75;
    int imageheightSize = 100;
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
            picture = Bitmap.createScaledBitmap(picture, imagewidthSize, imageheightSize, false);
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
        confidence = findViewById(R.id.confidence);

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
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 75, 100, 3}, DataType.FLOAT32);
            //allocateing the size to bytebuffer, 4 is bitmap float, 3 is rgb.
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imagewidthSize * imageheightSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imagewidthSize * imageheightSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            //for loop to go over the full pixels.
            //track the pixel number where we are on
            int pixel = 0;
            //iterate over each pixel and extract RGB values, add those values individually to the bytebuffer.
            for (int i = 0; i < imageheightSize; i++) {
                for (int j = 0; j < imagewidthSize; j++) {
                    int val = intValues[pixel++]; //RGB
//                    byteBuffer.putFloat(((val>>16)& 0xFF)* (1.f/1));
//                    byteBuffer.putFloat(((val>>8)& 0xFF)* (1.f/1));
//                    byteBuffer.putFloat((val& 0xFF)* (1.f/1));
                    //rescale the layer size to get the range from 0-1, that why delete 255.
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            //output

            //

            float[] confidences = outputFeature0.getFloatArray();
            //find the index of the class with the highest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Melanocytic nevi", "Dermatofibroma", "Vascular lesions", "Melanoma", "Benign keratosis-like lesions", "Basal cell carcinoma", "Actinic keratoses"};
            result.setText(classes[maxPos]);

            String s = "";
            for (int i =0; i<classes.length; i++){
                s+= String.format("%s: %1f%%\n",classes[i],confidences[i]*100);
            }
            confidence.setText(s);
         // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
}