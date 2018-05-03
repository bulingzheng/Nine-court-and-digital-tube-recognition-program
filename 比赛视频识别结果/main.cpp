#include "functions.h"
#include "include/function.h"
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

const int Width = 800;      // 视频宽
const int Height = 600;     // 视频高

int MaxArea = 8000;         // 每个宫格轮廓的最大面积
int MinArea = 4000;         // 每个宫格轮廓的最小面积

Mat frame, gray_img, canny_img;
Mat element0, element1, element2;

int t1 = 200, t2 = 250;     // canny阈值
int password[5];            // 密码区（数码管）的 5 个数字
int nineNumber[9];          // 九宫格区的 9 个数字
float neighborDistance[9];  // kNN 识别每个数字与最近邻居的距离，值越小说明是该值的可能性越大
int errorCount;             // 识别错误数字的个数
Point2f TubePoints[9];      //各个九宫个左上角
Point2f PwPoint[1];         //密码区左上角

Rect passwordRect(0, 0, 200, 60);   // 九宫格区的 Rect
Mat pw_BGR,pw_gray, pw_bin;                // 密码区（数码管）的灰度图和二值图
bool foundNixieTubeArea = false;    // 是否发现数码管区
bool isEmpty = false;               // 宫格中是否有数字

string to_string(int &x);

int main(int argc, char** argv)
{
    element0 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    element1 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    element2 = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
    ofstream outfile;
    outfile.open("output_all.txt");

    /************************************************************************/
    /*                     初始化caffe数字识别（手写体）                   */
    /************************************************************************/
    string model_file = "lenet.prototxt";
    string trained_file = "lenet_iter_20000.caffemodel";
    string mean_file = "mean_116.binaryproto";
    string label_file = "label.txt";
    Classifier classifier(model_file, trained_file, mean_file, label_file);








    /************************************************************************/
    /*                         开始检测每一帧图像                             */
    /************************************************************************/

    //主程序的真正入口
    double t = (double)getTickCount();
    VideoCapture capture("output12.avi");
    VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30.0, Size(800, 600));
    while(capture.isOpened())
    {
//        string frameName="data/"+to_string(t)+".jpg";
//        string frameName2="1.jpg";
//        Mat frame = imread(frameName);
        Mat frame;
        capture>>frame;
        Mat frame_copy=frame.clone();
        if(frame.empty())
        {
            cout<<"frame is not be readed"<<endl;
        }
//        imshow("frame", frame);
//        waitKey();

        cvtColor(frame, gray_img, COLOR_BGR2GRAY);
        Canny(gray_img, canny_img, t1, t2);
        dilate(canny_img, canny_img, element0); // 膨胀
//        imshow("canny", canny_img);
//        waitKey();

        // 寻找所有轮廓
        vector<vector<Point> > contours0;       // 所有轮廓
        findContours(canny_img, contours0, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        vector<vector<Point> > contours1;                 // 面积在指定范围内的轮廓（九宫格区），由counters0->counters1
        vector<double> areas;
        for(size_t i = 0; i < contours0.size(); i++)     // 先用面积约束，面积在指定范围内的轮廓的数量不满足要求时，进入下一帧
        {
            double area = contourArea(contours0[i]);
            areas.push_back(area);
            if(area > MinArea && area < MaxArea)
            {
                contours1.push_back(contours0[i]);
            }
        }

        if(contours1.size() < 9)                         //当满足面积要求的轮廓数量小于9个时，输出所有面积大于1000的查看一下
        {
            cout << "面积在指定范围内的轮廓数量不足" << endl;
            for(size_t i = 0; i < areas.size(); i++)
                if(areas[i] > 1000)
                    cout << areas[i] << " ";
            cout << endl;
        }

        // 上面得到的轮廓可能是有凸缺陷的（如果数字的笔画延伸到了轮廓的边缘）
        // 下面需要寻找凸包，绘制包含凸缺陷的较规范的矩形
        Mat contours_Mat(Height, Width, CV_8UC1, Scalar(0));
        for(size_t i = 0; i < contours1.size(); i++)
        {
            vector<int> hull;                                    // 找到的凸包(九宫格)（其实是轮廓最外层点的索引）
            convexHull(contours1[i], hull, true);                // 寻找点集（轮廓）的凸包
            int hullcount = (int)hull.size();                    //hull是凸包对应外缘点的编号值，hullcount是外缘点的数量
            Point point0 = contours1[i][hull[hullcount - 1]];    //找到凸包的起始点
            for(int j = 0; j < hullcount; j++)                   //在contours_Mat上画出凸包，以point0为起点
            {
                Point point = contours1[i][hull[j]];
                line(contours_Mat, point0, point, Scalar(255), 1, LINE_AA);
                point0 = point;
            }
        }
//        imshow("contours_Mat", contours_Mat);                                               //展示画出来的凸包（九宫格）
        vector<vector<Point> > contours2;                                                   //在查找凸包上的轮廓
        findContours(contours_Mat, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);    //查找到的轮廓保存在counters2中


        vector<RotatedRect> contours_rotatedRect;                                  //对面积在指定范围内的轮廓取最小包围矩形（旋转矩形）（九宫格）
        for(int i = 0; i < contours2.size(); i++)
        {
            if(contours2[i].size() < 100)                                          //对轮廓的数量先判断，轮廓数量小于100时，以最小包围矩形(带旋转角)包围countours2
            {
                contours_rotatedRect.push_back(minAreaRect(contours2[i]));
            }

        }


        if(contours_rotatedRect.size() > 9)                              // 如果最小包围矩形（带旋转角）数量大于9，则做一些约束，角度，长宽比
        {
            vector<RotatedRect> contours_rotatedRect_tmp;                //countours_rotatedRect_tmp只是个中间变量
            for(size_t i = 0; i < contours_rotatedRect.size(); i++)
            {
                bool b1 = false, b2 = false, b3 = false;

                float tmp_float = contours_rotatedRect[i].size.width / contours_rotatedRect[i].size.height;
                float tmp_area = contours_rotatedRect[i].size.width * contours_rotatedRect[i].size.height;
                if(tmp_float < 1)                                                                        //如果宽长比大小于1,说明长宽弄错了，定义新的长宽，新的倾斜角
                {
                    tmp_float = 1.0 / tmp_float;
                    swap(contours_rotatedRect[i].size.height, contours_rotatedRect[i].size.width);
                    contours_rotatedRect[i].angle = contours_rotatedRect[i].angle + 90;
                }

                if(contours_rotatedRect[i].angle > -5 && contours_rotatedRect[i].angle < 5)
                {
                    b1 = true;
                }

                if(tmp_float > 1.40 && tmp_float < 2.0)
                {
                    b2 = true;
                }

                if(tmp_area > MinArea && tmp_area < (MaxArea + 200))
                {
                    b3 = true;
                }

                if(b1 && b2)
                    contours_rotatedRect_tmp.push_back(contours_rotatedRect[i]);
            }
            contours_rotatedRect.clear();                                          //删除旧的最小包围矩形
            contours_rotatedRect = contours_rotatedRect_tmp;                       //将满足条件的最小包围矩形，保存到原有的contourd_rotatedRect变量当中
        }

        //找到了九个宫格的位置，提取对应ROI，用于识别
        vector<Mat>distimages(9);        //没有经过任何变换的九宫格数字
        vector<Mat> nineRect_mat(9);    // 经过阈值分割和膨胀处理的九宫格（数字）的Mat
        isEmpty = false;
        if(contours_rotatedRect.size() == 9)
        {
            sortRotatedRect(contours_rotatedRect);       // 把9个旋转矩形按照顺序排序
            for(int i = 0; i < 9; i++)                   // 对于每个旋转矩形，找到其四个顶点，使用仿射变换将其变换为正矩形,这样也就在一定程度上适应倾斜角
            {
                vector<Point2f> p(4);
                contours_rotatedRect[i].points(p.data());
                sortPoints(p);                           //获得旋转矩形的四个点

                Point2f srcPoints[4];
                Point2f dstPoints[4];


                srcPoints[0] = p[0] + Point2f(10, 8);    //旋转矩形的四个点往里收缩一点
                srcPoints[1] = p[1] + Point2f(-10, 8);
                srcPoints[2] = p[2] + Point2f(-10, -4);
                srcPoints[3] = p[3] + Point2f(10, -4);
                TubePoints[i]=srcPoints[0];              //依次获得各个九宫个数字的左上角点
                for(int j = 0; j < 4; j++)
                {
                    line(frame, srcPoints[j], srcPoints[(j + 1) % 4], Scalar(204, 122, 0), 2, LINE_AA); //画出缩小的四点连线
                }

                if(i == 0)   // 利用第一个宫格的右上角和第三个宫格的左上角来定位密码区（数码管）的位置，得到相应的Rect
                {
                    passwordRect.x = p[1].x - 20;
                    passwordRect.y = p[1].y - 80;
                    PwPoint[0].x=p[1].x - 20;       //获得密码区左上角点的位置
                    PwPoint[0].y=p[1].y-80;
                }
                else if(i == 2)
                {
                    passwordRect.width = p[0].x + 15 - passwordRect.x;
                    passwordRect.height = p[0].y - 20 - passwordRect.y;
                }

                if(passwordRect.x < 0 || (passwordRect.x + passwordRect.width) > Width ||
                        passwordRect.y < 0 || (passwordRect.y + passwordRect.height) > Height)   //如果密码区左上角超出图像范围，或者长宽超出范围，则认为没有发现密码区
                {
                    foundNixieTubeArea = false;
                }
                else                                                                            //否则认为找到密码区
                {
                    foundNixieTubeArea = true;
                }

                Point2f pwPoints[4];                    //将密码区用方框框起来
                int WIDTH=passwordRect.width;
                int HEIGHT=passwordRect.height;
                pwPoints[0].x=passwordRect.x;
                pwPoints[0].y=passwordRect.y;
                pwPoints[1]=pwPoints[0]+Point2f(WIDTH,0);
                pwPoints[2]=pwPoints[0]+Point2f(WIDTH,HEIGHT);
                pwPoints[3]=pwPoints[0]+Point2f(0,HEIGHT);
                for(int j = 0; j < 4; j++)
                {
                    line(frame, pwPoints[j], pwPoints[(j + 1) % 4], Scalar(204, 122, 0), 2, LINE_AA); //画出缩小的四点连线
                }



                dstPoints[0] = Point2f(0, 0);
                dstPoints[1] = Point2f(40, 0);
                dstPoints[2] = Point2f(40, 40);
                dstPoints[3] = Point2f(0, 40);

                Mat warpMat(2, 4, CV_32FC1);
                warpMat = getPerspectiveTransform(srcPoints, dstPoints);        //获得透视变换的矩形,包含原图初始四个点位置

                Mat dstImage(40, 40, CV_8UC1, Scalar(0));                       // 透视变换，将密码区变换成40*40的Mat,dstImage就是变成40×40的灰度图
                warpPerspective(gray_img, dstImage, warpMat, dstImage.size());


                // 九宫格内的数字在两次变换之间有短暂时间没有内容（空白）
                // 这里通过最低灰度值来判断是否存在数字
                if(i == 0 && min_mat(dstImage) > 80)
                {
                    cout << "宫格内没有数字" << endl;
                    isEmpty = true;
                }

//            imshow("nineRect_mat"+to_string(i)+".png",dstImage);       //显示经过透视变换的九宫格图片
//            imwrite("nineRect_mat"+to_string(i)+".png",dstImage);
                distimages[i]=dstImage.clone();                                                //密码区40×40灰度图保存到向量组中
//            threshold(distimages[i], nineRect_mat[i], 0, 255, THRESH_OTSU);                //对九宫格自动二值化
//            imshow("nineRect_mat_threshold"+to_string(i)+".png",nineRect_mat[i]);        //显示经过自动二值化的密码区图片
//            imwrite("nineRect_mat_threshold"+to_string(i)+".png",nineRect_mat[i]);
//            threshold(nineRect_mat[i], nineRect_mat[i], 50, 255, THRESH_BINARY_INV);       //反二值化操作
//            imshow("nineRect_mat_threshold_INV"+to_string(i)+".png",nineRect_mat[i]);    //反二值化
//            imwrite("nineRect_mat_threshold_INV"+to_string(i)+".png",nineRect_mat[i]);
//            dilate(nineRect_mat[i], nineRect_mat[i], element0);     // 膨胀
//            deskew(nineRect_mat[i]);    // 抗扭斜处理
//            blur(nineRect_mat[i], nineRect_mat[i], Size(3, 3));
//            imshow("nineRect_mat_threshold_INV_dilate"+to_string(i)+".png",nineRect_mat[i]);    //反二值化
//            imwrite("nineRect_mat_threshold_INV_dilate"+to_string(i)+".png",nineRect_mat[i]);
            }
        }
        else
        {
            cout<<"经过筛选后轮廓数量小于9"<<endl;
            continue;
        }
        // 如果第一个宫格中没有数字，则跳过该帧
        if(isEmpty)
        {
            cout<<"there is no number in the first tube"<<endl;
        }
        vector<Mat> nineMiddle50_mat(9);
        for(int i=0; i<9; i++)                                                             //对9个手写体进行常规处理
        {
            threshold(distimages[i], nineRect_mat[i], 0, 255, THRESH_OTSU);                //对九宫格自动二值化
            threshold(nineRect_mat[i], nineRect_mat[i], 50, 255, THRESH_BINARY_INV);       //反二值化操作
            dilate(nineRect_mat[i], nineRect_mat[i], element0);     // 膨胀
            deskew(nineRect_mat[i]);    // 抗扭斜处理
            blur(nineRect_mat[i], nineRect_mat[i], Size(3, 3));

            vector<vector<Point> >contours_1;                                                  //查找出轮廓,这一部分是将数字移动到中间位置,大小设置为50×50，保存在nineMiddle_mat
            findContours(nineRect_mat[i], contours_1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            vector<Point> Points_1;                                                            //一个轮廓就是一个点集，找到面积最大的那个点集，
            vector<vector<Point> >contours_2;                                                  //保存到点集向量contours_2
            Points_1 = contours_1[0];                                                          //从第一个轮廓点集开始
            for(size_t j = 0; j < contours_1.size(); j++)
            {
                if(contourArea(contours_1[j]) > contourArea(Points_1))
                {
                    Points_1 = contours_1[j];
                }
            }
            for(size_t k = 0; k < contours_1.size(); k++)
            {
                contours_2.push_back(Points_1);
            }
            Mat bd=Mat(Size(50,50), CV_8UC1, Scalar(0));     //制作最终显示用的底板
            Rect rect = boundingRect(contours_2[0]);   //根据点集获得包围矩形
            int width = rect.width;
            int height = rect.height;
            int x = 50 / 2 - width / 2;
            int y = 50 / 2 - height / 2;
            Rect rect_2;
            rect_2 = Rect(x, y, width, height);
            nineRect_mat[i](rect).copyTo(bd(rect_2));      //将图像数字位置图像信息，复制到底板nine_Middle_mat上
            nineMiddle50_mat[i]=bd.clone();
        }


        Mat nineNumberMat(150, 150, CV_8UC1);                                 //将9个数字按照顺序在一张图里展示
        for(int i = 0; i < 3; i++)                                            //i代表了列
            for(int j = 0; j < 3; j++)                                        //j代表了行
                nineMiddle50_mat[j + i * 3].copyTo(nineNumberMat(Rect(j * 50, i * 50, 50, 50)));
//        imshow("nineNumberMat", nineNumberMat);


        // 由数码管区的 passwordRect 得到相应的ROI，并做一些预处理
        pw_gray = gray_img(passwordRect);
        threshold(pw_gray, pw_bin, 200, 255, THRESH_BINARY);
        erode(pw_bin, pw_bin, element2);        // 腐蚀
        dilate(pw_bin, pw_bin, element1);       // 膨胀
        connectClosedPoint(pw_bin);             //连接相近像素点
//        imshow("password", pw_bin);             //经过二值化和腐蚀膨胀后的密码图

        Mat pw_bin_ = pw_bin.clone();
        vector<vector<Point> > ninxiTubeAreaContour;
        findContours(pw_bin_, ninxiTubeAreaContour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);     //查找几个数码管的轮廓

        vector<Rect> ninxiTubeNumbRect;
        for(size_t i = 0; i < ninxiTubeAreaContour.size(); i++)
        {
            Rect tmpRect = boundingRect(ninxiTubeAreaContour[i]);       //对数码管轮廓标准包围矩形tmpRect
            int tmpHeight = MAX(tmpRect.width, tmpRect.height);
            if(tmpHeight > 20)
                ninxiTubeNumbRect.push_back(tmpRect);
        }



        if(ninxiTubeNumbRect.size() == 5)        //如果有5个包围矩形
        {
            sortRect(ninxiTubeNumbRect);        //对密码区几个数字进行先后排序
        }
        else
        {
            cout << "数码管识别错误，ninxiTubeNumbRect.size() = " << ninxiTubeNumbRect.size() << endl;
            waitKey(10);

        }

        /************************************************************************/
        /*                     caffe识别密码区（数码管）数字                        */
        /************************************************************************/
//        cout<<frameName<<" ";       //输出读取图片的名字
//        outfile<<frameName<<" ";
        vector<Mat> matPw(5);
        for(int i = 0; i < 5; i++)     //密码区数字居中调整成正方形
        {
            Mat matbd= pw_bin(ninxiTubeNumbRect[i]);
            int MaxSize = MAX(matbd.rows, matbd.cols);
            Mat dst(MaxSize, MaxSize, CV_8UC1, Scalar(0));
            Rect ROI = Rect((MaxSize - matbd.cols) / 2, (MaxSize - matbd.rows) / 2, matbd.cols, matbd.rows);
            matbd.copyTo(dst(ROI));
            matPw[i]=dst.clone();
        }

        for(int i=0; i<5; i++)
        {
            Mat matROI =matPw[i].clone();
            resize(matROI, matROI, Size(40, 40));
            std::vector<Prediction> predictions = classifier.Classify(matROI);
            cout<<predictions[0].first<<"|";
            outfile<<predictions[0].first<<"|";
            string number=predictions[0].first+"|";
            Point2f point_plus=PwPoint[0]+Point2f(i*40,-10);
            putText(frame,number,point_plus,CV_FONT_HERSHEY_SIMPLEX,0.8,Scalar(0,0,255),2,2); // 显示字符

        }
        cout<<"-->";
        outfile<<"-->";

        /************************************************************************/
        /*                         caffe识别九宫格区数字                          */
        /************************************************************************/
        for(int i=0; i<9; i++)
        {
            std::vector<Prediction> predictions = classifier.Classify(nineMiddle50_mat[i]);
            cout<<predictions[0].first;
            outfile<<predictions[0].first;
            string number=predictions[0].first;
            putText(frame,number,TubePoints[i],CV_FONT_HERSHEY_SIMPLEX,0.8,Scalar(0,0,255),2,2);

        }
        cout << endl;
        outfile<<endl;
        imshow("frame", frame);
//        string frameName_write="detection-result/"+to_string(t)+".jpg";
//        imwrite(frameName_write,frame);
        writer<<frame;
        waitKey(30);
    }
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout<<"t: "<<t<<endl;



//    waitKey(0);


    return 0;
}

string to_string(int &x)
{
    stringstream ss;
    string y;
    ss << x;
    ss >> y;
    return y;
}
/*
double time0 = static_cast<double>(getTickCount());
time0 = ((double)getTickCount() - time0) / getTickFrequency();
cout << "用时" << time0 * 1000 << "毫秒" << endl;
*/

