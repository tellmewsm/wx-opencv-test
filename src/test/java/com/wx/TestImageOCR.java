package com.wx;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;
import org.testng.annotations.Test;

import java.text.DecimalFormat;
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

    static { System.load(System.getProperty("user.dir") + "/src/main/resources/opencv_java342.dylib");}

    private List<Integer> testLabs = new ArrayList<>();
    private StringBuffer result=new StringBuffer();

    @Test
    public void TestImageKnn() {

        Mat mat = Imgcodecs.imread("src/main/resources/numbers/upper_case_letter_1.jpg");

        Mat Data = getImageTrain(mat);

        KNearest knn = KNearest.create();

        Map testMap=getImageLabs();

        //训练数据，每行一个样本，第三个参数为训练样本对应的正确答案
        knn.train(Data, Ml.ROW_SAMPLE, Converters.vector_int_to_Mat(testLabs));

        //测试数据
        Mat testData = Imgcodecs.imread("src/main/resources/numbers/upper_case_letter_2.jpg");
        Mat imageTrain = getImageTrain(testData);
        int err = 0;

        for (int i = 0; i < testMap.keySet().size(); i++) {
            // 读取一行
            Mat one_feature = imageTrain.row(i);

            //Imgcodecs.imwrite("/Users/wuxi/Pictures/testdata/" + "testData" + one_feature + ".jpg", one_feature);

            // 预期答案
            int testLabel = testLabs.get(i);
            System.out.println("testLabel============" + testMap.get(testLabel));

            Mat res = new Mat();

            // 查找匹配：第一个参数为输入样本（可一次输入多个样本），第二个参数为需要返回的K个邻近（即KNearest的那个K），第三个参数为返回结果（res: result），结果为样本对应的Label值，每一个样本的匹配结果对应一行
            // 如果输入仅有一个样本，则返回结果（p）就是预测结果。参数 1即为K-近邻算法的关键参数
            float p = knn.findNearest(one_feature, 1, res);

            System.out.println("result============"+ testMap.get((int)(p)) + " " + res.dump());
            result.append(testMap.get((int)(p)));

            int iRes = (int) p;
            if (iRes != testLabel) {
                err++;
            }
        }
        System.out.println("识别结果============"+result);
        float accuracy = (float) ((testMap.keySet().size() - (float)err) / testMap.keySet().size());
        DecimalFormat df = new DecimalFormat("0.0000");
        System.out.println("error count: " + err + ", accuracy is: " + df.format(accuracy));

    }


    //获取训练数据
    public Mat getImageTrain(Mat mat) {

        Mat trainData = new Mat();

        Mat dst = mat.clone();
        //灰度
        Imgproc.cvtColor(mat, dst, Imgproc.COLOR_BGR2GRAY);
        //二值化
        Imgproc.threshold(dst, dst, 245, 255, 1);

        List<MatOfPoint> contours = new ArrayList<>();

        //在二值化图像中查找轮廓,最外层
        Imgproc.findContours(dst, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint cnt : contours) {

            Rect rect = Imgproc.boundingRect(cnt);

            //计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的，points 读入的参数必须是vector或者Mat点集
            int x = rect.x;
            int y = rect.y;
            int w = rect.width;
            int h = rect.height;

            if (w>6&&h>10) {

                Imgproc.rectangle(mat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255), 1); //线的宽度

                //HighGui.imshow("mat",mat);
                //HighGui.waitKey(50);

                //输出图片查看
                Rect image = new Rect(x, y, w, h);
                //裁剪区域
                Mat imgROI = new Mat(dst, image);

                //无需轮廓手动切割
                //Mat num = mat.submat(new Rect(20, 40, 20, 20));
                //缩放
                Imgproc.resize(imgROI, imgROI, new Size(20, 40));
                // knn算法的输入为浮点型，在此转换
                imgROI.convertTo(imgROI, CvType.CV_32F);

                trainData.push_back(imgROI.reshape(1, 1));
            }
        }

        //Imgcodecs.imwrite("/Users/wuxi/Pictures/picture/juxing.jpg", mat);

        return trainData;
    }

    //获取训练结果
    public Map<Integer, Character> getImageLabs() {

        Map<Integer, Character> testMap=new HashMap<>();

        String test = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        for (int i = 0; i < test.length(); i++) {

            char fir = test.charAt(i);

            testLabs.add(i);

            testMap.put(i,fir);
        }

        return testMap;
    }


}
