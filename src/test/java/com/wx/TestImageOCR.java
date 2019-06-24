package com.wx;

import com.alibaba.fastjson.JSONObject;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @ClassName TestImageOCR
 * @Description //TODO
 * @Author wuxi
 * @Date 2019-06-24 22:01
 * @Version 1.0
 **/
public class TestImageOCR {

    static{ System.load("/Users/wuxi/Documents/opencv/opencv_java342.dylib");}

    @Test
    public void TestImageTrain(Mat mat) {

        Mat dst = mat.clone();
        //灰度
        Imgproc.cvtColor(mat, dst, Imgproc.COLOR_BGR2GRAY);
        //二值化
        Imgproc.threshold(dst, dst,245,255,1);

        Imgcodecs.imwrite("/Users/wuxi/Documents/opencv/number_1_ocr.jpg", dst);

        List<MatOfPoint> contours = new ArrayList<>();

        //在二值化图像中查找轮廓,最外层
        Imgproc.findContours(dst,contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);

        JSONObject jsonObject = new JSONObject();
        List<Map> list=new ArrayList<>();
        for (MatOfPoint cnt : contours) {

            Rect rect = Imgproc.boundingRect(cnt);

            //计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的，points 读入的参数必须是vector或者Mat点集
            int x = rect.x;
            int y = rect.y;
            int w = rect.width;
            int h = rect.height;

            Imgproc.rectangle(dst, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),new Scalar(0, 0, 255), 1); //线的宽度
            //输出图片查看
            Imgcodecs.imwrite("/Users/wuxi/Desktop/juxing.jpg",dst);

            Rect image=new Rect(x, y, w, h);
            //裁剪区域
            Mat imgROI = new Mat(dst, image);
            //缩放
            Imgproc.resize(imgROI, imgROI, new Size(20, 40));

            Imgcodecs.imwrite("/Users/wuxi/Desktop/juxing"+rect+".jpg",imgROI);

            jsonObject.fluentPut("location",list);
        }
    }

    @Test
    public void TestImageOcr() {

        Mat mat= Imgcodecs.imread("/Users/wuxi/Downloads/WechatIMG1.jpeg");
//        Mat src= Imgcodecs.imread("/Users/wuxi/Downloads/number_1.jpg");
//
//        TestImageTrain(src);

        TestImageTrain(mat);

    }
}
