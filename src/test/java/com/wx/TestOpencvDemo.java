package com.wx;

import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;
import org.testng.annotations.Test;

/**
 * Created by wuxi on 2019/6/20.
 */
public class TestOpencvDemo {

    static{ System.load("/Users/wuxi/Downloads/opencv_java342.dylib");}

    public static void main(String[] args) {

            Mat src=Imgcodecs.imread("/Users/wuxi/Desktop/test.jpeg");

            Mat dst = src.clone();

            Imgproc.blur(src,dst,new Size(30,30));

            Imgcodecs.imwrite("/Users/wuxi/Desktop/test1.jpeg", dst);

    }

    @Test
    public void TestMat() {

        Mat img = new Mat(3, 3, CvType.CV_8UC3);//new Scalar(0,255,0)

        Mat dist = img.clone();

        img.put(0,0,new byte[]{0, (byte) 255,0,(byte) 255,0,0,0,0,(byte) 255});

        System.out.println(img.size());//矩阵大小 3X3

        Imgproc.resize(img,dist,new Size(6,6));

        Imgcodecs.imwrite("/Users/wuxi/Desktop/img2.jpeg", img);

        Imgcodecs.imwrite("/Users/wuxi/Desktop/img3.jpeg", dist);

        System.out.println(img.dims()+"维"+"\n"+img.channels()+"通道"+"\n"+img.dump());

        System.out.println(dist.size());

        System.out.println(dist.dims()+"维"+"\n"+img.channels()+"通道"+"\n"+dist.dump());

        img.release();

        dist.release();

    }

    @Test
    public void TestMatRead() {

        Mat originalImg = Imgcodecs.imread("/Users/wuxi/Desktop/img2.jpeg", 0);

        System.out.println(originalImg.dump());

        Imgcodecs.imwrite("/Users/wuxi/Desktop/"+"img2img2"+".jpg",originalImg);

    }
}
