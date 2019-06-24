package com.wx;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by PokerDealer on 2019/5/23.
 */
public class TestHsvLocation {

    static{ System.load("/Users/wuxi/Downloads/opencv_java342.dylib");}

    public static void main(String[] args) {

        JSONArray jsonArray = areaCleanLocation("/Users/wuxi/Downloads/image001.png").getJSONArray("location");

        for (int i = 0; i < jsonArray.size(); i++) {
            System.out.println(jsonArray.getJSONObject(i).getInteger("x"));
            System.out.println(jsonArray.getJSONObject(i).getInteger("y"));
        }
    }

    public static JSONObject areaCleanLocation(String originalImgPath){

        //设定颜色HSV范围，假定图片中特定的颜色
        Scalar lower = new Scalar(171,168,167);
        Scalar upper = new Scalar(174,174,174);

        //读取图像 Mat是OpenCV中用来存储图像信息的内存对象，可以理解为一个包含所有强度值的像素点矩阵，另外包含其他信息（宽，高，类型，纬度，大小，深度等）
        Mat img = Imgcodecs.imread(originalImgPath);
        //克隆图像mat,outputarry
        Mat dstImage = img.clone();

        //中值滤波操作,中值滤波是基于排序统计理论的一种能有效抑制噪声的非线性信号处理技术，对脉冲噪声有良好的滤除作用，特别是在滤除噪声的同时，能够保护信号的边缘，使之不被模糊
        //OutputArray 目标图像
        //int ksize	孔径线性尺寸：必须是大于1的奇数，例如3、5、7,参数越大越模糊
        Imgproc.medianBlur(img, dstImage, 7);

        //输出图片查看
        Imgcodecs.imwrite("/Users/wuxi/Desktop/medianBlur.jpg",dstImage);

        Mat mask = new Mat();
        //void inRange(InputArray src, InputArray lowerb, InputArray upperb, OutputArray dst)
        //即检查数组元素是否在另外两个数组元素值之间
        //这里的数组通常也就是矩阵Mat或向量
        //OpenCV-利用函数inRange进行颜色分割，HSV颜色
        Core.inRange(dstImage, lower, upper, mask);
        //输出图片查看，二值化图片，黑白
        Imgcodecs.imwrite("/Users/wuxi/Desktop/erzhihua.jpg",mask);

        List<MatOfPoint> contours = new ArrayList<>();
        //RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息，如果内孔内还有一个连通物体，这个物体的边界也在顶层
        //CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

        //在二值化图像中查找轮廓，RETR_CCOMP 可以好几层
        Imgproc.findContours(mask,contours,new Mat(),Imgproc.RETR_CCOMP,Imgproc.CHAIN_APPROX_SIMPLE);

        JSONObject jsonObject = new JSONObject();
        List<Map> list=new ArrayList<>();
        for (MatOfPoint cnt : contours) {

            Rect rect = Imgproc.boundingRect(cnt);

            //计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的，points 读入的参数必须是vector或者Mat点集
            int x = rect.x;
            int y = rect.y;
            int w = rect.width;
            int h = rect.height;
            Map<String, Integer> map = new HashMap<String, Integer>();
            if (h/ w > 0.9 && h / w < 1.1) {
                map.put("x",x);
                map.put("y",y);
                list.add(map);
            }
            Imgproc.rectangle(dstImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),new Scalar(0, 255, 0), 2); //线的宽度
            //输出图片查看
            Imgcodecs.imwrite("/Users/wuxi/Desktop/juxing2.jpg",dstImage);

            jsonObject.fluentPut("location",list);
        }

        img.release();

        return jsonObject;
    }

}
