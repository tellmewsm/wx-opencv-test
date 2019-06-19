package com.wx;

import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;

public class TestBlur {

    static{ System.load(System.getProperty("user.dir")+"/src/main/resources/opencv_java342.dylib"); }

    public static void main(String[] args) {

            Mat src=Imgcodecs.imread("/Users/wuxi/Desktop/test.jpeg");
            Mat dst = src.clone();

            //图像模糊处理
            Imgproc.blur(src,dst,new Size(30,30));
            Imgcodecs.imwrite("/Users/wuxi/Desktop/test1.jpeg", dst);

    }

}
