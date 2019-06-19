package com.wx;

import com.alibaba.fastjson.JSONObject;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by PokerDealer on 2019/5/23
 */
public class TestMatch {

    static{ System.load(System.getProperty("user.dir")+"/src/main/resources/opencv_java342.dylib"); }

    public static void main(String[] args) {

        String basicImgPath = "/Users/wuxi/Downloads/M0.jpg";
        String templateImgPath = "/Users/wuxi/Downloads/M2.jpg";
        String originalImgPath = "/Users/wuxi/Downloads/3.jpg";

        System.out.println(isIconExist(basicImgPath, templateImgPath, originalImgPath));

    }

    public static boolean isIconExist(String basicImgPath, String templateImgPath, String originalImgPath) {

        Mat originalImg = Imgcodecs.imread(originalImgPath, 0);
        int originalImgHeight = originalImg.height();
        int originalImgWidth = originalImg.width();

        System.out.println(originalImgWidth);
        System.out.println(originalImgHeight);

        Mat templateImg = Imgcodecs.imread(templateImgPath, 0);
        Mat templateImgResize = templateImg.clone();
        int templateImgHeight = templateImg.height();
        int templateImgWidth = templateImg.width();

        Mat basicImg = Imgcodecs.imread(basicImgPath, 0);
        List<MatOfPoint> basicContours = findContours(basicImg);

        if (basicContours.size() != 1) {
            System.out.println("所选取的基础模板图片不符合规范");
        }

        List<MatOfPoint> originalContours = findContours(originalImg);

        double ratio = getRatio(basicContours, originalContours, originalImgWidth / 2);

        //修改图像大小Resize Imgproc.resize,将需要对比图像进行比例缩放 在进行对比
        Imgproc.resize(templateImg, templateImgResize, new Size(templateImgWidth * ratio, templateImgHeight * ratio));

        Imgcodecs.imwrite("/Users/wuxi/Desktop/M0_1Resize.jpg",templateImgResize);

//        HighGui.imshow("tt",dstImg);
//        HighGui.waitKey(0);
        Boolean result = matchImg(originalImg, templateImgResize);

        return result;
    }

    public static List<MatOfPoint> findContours(Mat img) {
        Mat dstImage = img.clone();

        //图像的模糊化Blur处理,输出 dstImage 3*3，平滑处理
        Imgproc.blur(img, dstImage, new Size(3, 3));

        Imgcodecs.imwrite("/Users/wuxi/Desktop/M0_1"+img+".jpg",dstImage);

        int dstChannels = dstImage.channels();
        Mat edges = img.clone();

        //边缘检测
        Imgproc.Canny(dstImage, edges, 40, 80);

        Imgcodecs.imwrite("/Users/wuxi/Desktop/M0_2"+img+".jpg",edges);

//        HighGui.imshow("tt",edges);
//        HighGui.waitKey(0);

        List<MatOfPoint> contours = new ArrayList<>();

        Mat hierarchy = new Mat();

        Imgproc.findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

 /*       for (MatOfPoint cnt : contours) {

            Rect rect = Imgproc.boundingRect(cnt);
            //计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的，points 读入的参数必须是vector或者Mat点集
            Imgproc.rectangle(edges, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),new Scalar(0, 255, 0), 2); //线的宽度
            //输出图片查看
            Imgcodecs.imwrite("/Users/wuxi/Desktop/juxing"+cnt+".jpg",edges);

        }*/

        return contours;
    }

    public static double getRatio(List<MatOfPoint> templateContours, List<MatOfPoint> originalContours, int halfWidth) {
        MatOfPoint templateCnt = templateContours.get(0);
        MatOfPoint2f points1 = new MatOfPoint2f(templateContours.get(0).toArray());

        Point center = new Point();
        float[] radius1 = {0};
        //表示输出的圆形的中心坐标，是float型
        Imgproc.minEnclosingCircle(points1, center, radius1);

        double value = 1;
        for (MatOfPoint originalCnt : originalContours) {
            if (Imgproc.matchShapes(originalCnt, templateCnt, 1, 0.0) < 0.001) {
                MatOfPoint2f point2f2 = new MatOfPoint2f(originalCnt.toArray());
                Point point2 = new Point();
                float[] radius2 = {0};

                //表示输出的圆形的中心坐标，是float型
                Imgproc.minEnclosingCircle(point2f2, point2, radius2);
                double x = point2.x;
                double y = point2.y;

                if (Math.abs(x - new Double(halfWidth)) / new Double(halfWidth) < 0.1) {
                    value = (double) (radius2[0] / radius1[0]);
                    System.out.println(radius2[0]);
                    System.out.println(radius1[0]);
                    System.out.println(value);
                }
            }
        }
        return value;
    }

    public static boolean matchImg(Mat originalImg, Mat templateImg) {
        Mat result = originalImg.clone();
        Imgproc.matchTemplate(originalImg, templateImg, result, Imgproc.TM_CCOEFF_NORMED);
        //两幅进行对比，归一化相关系数匹配法

        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(result);

        if (minMaxLocResult.maxVal > 0.9) {
            System.out.println("匹配成功");
            return true;
        } else {
            System.out.println("匹配失败");
            return false;
        }
    }

}
