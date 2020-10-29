package com.wx;

import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;
import org.testng.annotations.Test;

/**
 * @ClassName TestOpencvDemo
 * @Description //TODO
 * @Author wuxi
 * @Date 2019-06-20 20:00
 * @Version 1.0
 **/
public class TestOpencvDemo {

    static String path = System.getProperty("user.dir") + "/src/main/resources/";

    static {
        System.load(path + "opencv_java342.dylib");
    }

    public static void main(String[] args) {

            Mat src=Imgcodecs.imread(path+"/opencv.png");

            Mat dst = src.clone();

            Imgproc.medianBlur(dst, dst, 7);

            // Imgproc.blur(src,dst,new Size(30,30));

            Imgcodecs.imwrite(path+"/opencvdemo/opencv1.png", dst);

    }

    @Test
    public void TestMatCv() {

        //二值图像，0到255，0是黑色，255白色
        Mat img = new Mat(3, 3, CvType.CV_8UC1);

        img.put(0,0,new byte[]{0, (byte) 255,0});

        Imgcodecs.imwrite(path+"/opencvdemo/imgCV_8UC1.jpeg", img);

        //打印矩阵
        System.out.println(img.dump());

        img.release();

    }

    @Test
    public void TestMatResize() {

        Mat img = new Mat(3, 3, CvType.CV_8UC3);//new Scalar(0,255,0)

        Mat dist = img.clone();

        img.put(0,0,new byte[]{0, (byte) 255,0,(byte) 255,0,0,0,0,(byte) 255});

        //矩阵大小 3X3
        System.out.println(img.size());

        Imgproc.resize(img,dist,new Size(6,6));

        Imgcodecs.imwrite(path+"/opencvdemo/img2.jpeg", img);

        Imgcodecs.imwrite(path+"/opencvdemo/img3.jpeg", dist);

        System.out.println(img.dims()+"维"+"\n"+img.channels()+"通道"+"\n"+img.dump());

        System.out.println(dist.size());

        System.out.println(dist.dims()+"维"+"\n"+img.channels()+"通道"+"\n"+dist.dump());

        img.release();

        dist.release();

    }

    @Test
    public void TestMatReshape() {

        Mat img = new Mat(3, 3, CvType.CV_8UC3,new Scalar(0,255,0));

        System.out.println(img.dump());

        //拉扯成一行
        System.out.println(img.reshape(1,1).dump());

    }


}
