package com.example.imageclassification;

import androidx.annotation.Nullable;
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

//import com.example.imageclassification.ml.Model0;;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result;
    Button gallery,camera;
    int imagewidthSize =75;
    int imageheightSize = 100;
    ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {

            @TargetApi(Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {

//                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(cameraIntent,3);
                    }else{
                        requestPermissions(new String[]{Manifest.permission.CAMERA},100);
                    }
                }
//            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent,1);
            }
        });
    }

    ////cancel annotation here!!!!
//    public void classifyImage(Bitmap image){
//        try {
//            Model0 model = Model0.newInstance(getApplicationContext());
//
//            // Creates inputs for reference.
//            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 75, 100, 3}, DataType.FLOAT32);
//            //allocateing the size to bytebuffer, 4 is bitmap float, 3 is rgb.
//            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4* imagewidthSize * imageheightSize *3);
//            byteBuffer.order(ByteOrder.nativeOrder());
//
//            int[]  intValues = new int[imagewidthSize*imageheightSize];
//            image.getPixels(intValues,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
//            //for loop to go over the full pixels.
//            //track the pixel number where we are on
//            int pixel = 0;
//            //iterate over each pixel and extract RGB values, add those values individually to the bytebuffer.
//            for(int  i =0; i < imageheightSize; i ++ ){
//                for (int j=0; j<imagewidthSize; j++){
//                    int val = intValues[pixel++]; //RGB
////                    byteBuffer.putFloat(((val>>16)& 0xFF)* (1.f/1));
////                    byteBuffer.putFloat(((val>>8)& 0xFF)* (1.f/1));
////                    byteBuffer.putFloat((val& 0xFF)* (1.f/1));
//                    //rescale the layer size to get the range from 0-1, that why delete 255.
//                    byteBuffer.putFloat(((val>>16)& 0xFF)* (1.f/255));
//                    byteBuffer.putFloat(((val>>8)& 0xFF)* (1.f/255));
//                    byteBuffer.putFloat((val& 0xFF)* (1.f/255));
//                }
//            }
//
//            inputFeature0.loadBuffer(byteBuffer);
//
//            // Runs model inference and gets result.
//            Model0.Outputs outputs = model.process(inputFeature0);
//            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//            //output
//
//            float[] confidences = outputFeature0.getFloatArray();
//            //find the index of the class with the highest confidence.
//            int maxPos = 0;
//            float maxConfidence = 0;
//            for (int i=0; i< confidences.length;i++){
//                if (confidences[i]>maxConfidence){
//                    maxConfidence= confidences[i];
//                    maxPos=i;
//                }
//            }
//            String[] classes = {"Melanocytic nevi","Dermatofibroma","Vascular lesions","Melanoma","Benign keratosis-like lesions","Basal cell carcinoma","Actinic keratoses"};
//            result.setText(classes[maxPos]);
//
//            // Releases model resources if no longer used.
//            model.close();
//        } catch (IOException e) {
//            // TODO Handle the exception
//        }
//    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == RESULT_OK){
            // check if the image gets from camera,as requestcode is 3
            if (requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("Data");
                //because the CNN trained the pictures in squares, so resize the image.
                //change image size to dimension size, dimension is the min size from the image.
//                int dimension = Math.min(image.getWidth(),image.getHeight());
//                //recale the image and make it square.
//                image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
                image = ThumbnailUtils.extractThumbnail(image,image.getWidth(),image.getHeight());
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image,imagewidthSize,imageheightSize,false);
//                showImage(image);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                        image = MediaStore.Images.Media.getBitmap(this.getContentResolver(),dat);
                }catch (IOException e){
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image,imagewidthSize,imageheightSize,false);
                classifyImage(image);
//                showImage(image);
            }


        }
        super.onActivityResult(requestCode, resultCode, data);
    }

//    private void showImage(Bitmap image) {
//        imageView.setImageBitmap(BitmapFactory.decodeFile("pathToImageFile"));
//    }
}