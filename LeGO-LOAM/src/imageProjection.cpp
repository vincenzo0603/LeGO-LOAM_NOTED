// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

/**
 * @file imageProjection.cpp
 * 这是lego-loam的第一步，第一个节点用于分割scan点云：
 * 将输入点云投射到深度图像：
 *                       1.对深度图像进行列的评估作为地面平面估计，找到地面平面点（地面平面点不参与分割），并给出标签;
 *                       2.用基于图像的分割方法对深度图非地面平面点进行分割，将点聚集成许多cluster（集群，类），同一个cluster具有独一的标签（目的是为了快速可靠的特征提取）
 * imageProjection处理之后获得三个属性：
 *                       1.地面平面点和分割点的标签
 *                       2.点在深度图的行和列的索引
 *                       3.点的深度值
 * 这三个属性将在后面模块用到
 */
#include "utility.h"

class ImageProjection{
private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;       // 接收到的来自激光Msg的原始点云数据
    pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;  // 用 laserCloudInRing 存储含有具有通道R的原始点云数据

    pcl::PointCloud<PointType>::Ptr fullCloud;          // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud;      // same as fullCloud, but with intensity - range

    // 所有点分为被分割点、未被分割点、地面点、无效点。
    pcl::PointCloud<PointType>::Ptr groundCloud;        // 地面点点云
    pcl::PointCloud<PointType>::Ptr segmentedCloud;     // segMsg 点云数据:包含被分割点和经过降采样的地面点
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure; // 存储被分割点点云，且每个点的i值为分割标志值
    pcl::PointCloud<PointType>::Ptr outlierCloud;       // 经过降采样的未分割点

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat; // FLT_MAX： 初始化值； range： 点云的深度值。 range matrix for range image
    cv::Mat labelMat; // =-1：无效值，不进行点云分割；  =0： 初始值；  =labelCount：  平面点；=999999： 舍弃点。 label matrix for segmentaiton marking
    cv::Mat groundMat; //=-1：无效值，表示无法判断是否为地面点；=0： 代表初始值，不是地面点； =1： 表示是地面点。 ground matrix for ground cloud marking
    int labelCount;

    float startOrientation;
    float endOrientation;
    
    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

public:
    ImageProjection():
        //nh("~")节点句柄，为节点/话题命名
        nh("~"){
        // 订阅sensor_msgs::PointCloud2 而不是 sensor_msgs::LaserScan 是激光雷达的原因吗 ？velodyne输出"/velodyne_points"(sensor_msgs::PointCloud2) 
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1); // featureAssociation中有订阅此话题："/segmented_cloud"
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1); // cloud_info 自定义消息类型，featureAssociation中有订阅此话题："/segmented_cloud_info"
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);     // featureAssociation中有订阅此话题："/outlier_cloud"

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();   // 分配内存和参数初始化
        resetParameters();  // 初始化 / 重置各类参数内容，为下一次循环做准备
    }

    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor); // 下
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor); // 右
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor); // 左
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor); // 上

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection(){}

    /**
     * @brief 将ros中的PointCloud2ConstPtr& laserCloudMsg 转为pcl点云库指针
     * 
     * @param laserCloudMsg 
     */
    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        
        cloudHeader = laserCloudMsg->header;
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
        // have "ring" channel in the cloud   如果useCloudRing = true, 则不使用 ang_res_y 和 ang_bottom 
        if (useCloudRing == true){
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
            if (laserCloudInRing->is_dense == false) {  // 这里判断是否为is_dense 但是并没有提前remove nan值？
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }  
        }
    }
    
    /**
     * @brief cloudHandler 是 imageProjection.cpp 中最主要的一个函数,由它调用其他的函数：
     * 1.复制点云数据 2.找到开始和结束的角度 3.移除地面点 4.点云分块 5.发布处理后的点云数据 6.重置参数
     * 
     * @param laserCloudMsg 
     */
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);
        // 2. Start and end angle of a scan
        findStartEndAngle();
        // 3. Range image projection
        projectPointCloud();
        // 4. Mark ground points
        groundRemoval();
        // 5. Point cloud segmentation
        cloudSegmentation();
        // 6. Publish all clouds
        publishCloud();
        // 7. Reset parameters for next iteration
        resetParameters();
    }

    void findStartEndAngle(){
        // start and end orientation of this cloud
        // atan2(...)负数的原因是：雷达旋转方向和坐标系定义下的右手定则正方向不一致
        // 我们一般认为θ_start 属于[-pi,pi],而θ_end 属于[pi,3pi],而两者的角度差不会与2π相差太多
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                                     laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

    /**
     * range image(16*1800)
     * ==========================================================================================================================
     *                      列索引:columnIdn(degree) 
     *                      0(0.0) 1(0.2) 2(0.4)
     * 行索引:rowIdn(degree)   
     *              0(-15)
     *              1(-13)
     *               ...
     *              14(13)
     *              15(15)
     * ==========================================================================================================================
     */

    void projectPointCloud(){
        // range image projection
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();    // 接收原始点云个数

        for (size_t i = 0; i < cloudSize; ++i){

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            // 找到这个点在image中的row和column索引     find the row and column index in the iamge for this point
            if (useCloudRing == true){
                rowIdn = laserCloudInRing->points[i].ring;
            }
            else{
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;      // verticalAngle是此点在激光雷达坐标系中的垂直方向的角度，加ang_bottom是为了消除-15度的影响
            }                                                           // 总体 除以角度分辨率(ang_res_y)得到rowIdn
            if (rowIdn < 0 || rowIdn >= N_SCAN)                         // 出现异常角度则无视它
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;// 计算以y轴正方向的基准的水平角度

            // 这里认为horizonAngle-90.0 是激光雷达坐标系转为普通坐标系
            //          ^
            //          | x
            //          |
            // (y)<----------
            //          |
            //          |
            //转为：
            //          ^
            //          | y
            //          |
            //     ----------->x
            //          |
            //          |
            // 下面columnIdn和两个if代码：先把columnIdn从horizonAngle:(-PI,PI]转换到columnIdn:[H/4,5H/4],然后判断columnIdn大小，
            // 再将它的范围转换到了[0,H] (H:Horizon_SCAN)。
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;     
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);    // 这里存储的是欧式距离而非垂直距离
            if (range < sensorMinimumRange)
                continue;
            
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;       // 为什么intensity = (float)rowIdn + (float)columnIdn / 10000.0 ？？？？？

            index = columnIdn  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"
        }
    }


    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat
        // -1, no valid info to check if ground of not
        //  0, initial value, after validation, means not ground
        //  1, ground
        // 两个for循环遍历fullcloud，fullcloud的索引与image range索引是一致的，所以可以理解为固定列j，遍历行i(i < 7)
        // 然后判别点是否为ground点
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            for (size_t i = 0; i < groundScanInd; ++i){

                lowerInd = j + ( i )*Horizon_SCAN;
                upperInd = j + (i+1)*Horizon_SCAN;

                // 初始化的时候用nanPoint.intensity = -1 填充
                // 都是-1 证明是空点nanPoint
                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }

                // 由上下两线之间点的XYZ位置得到两线之间的俯仰角
				// 如果俯仰角在10度以内，则判定(i,j)为地面点,groundMat[i][j]=1
				// 否则，则不是地面点，进行后续操作 
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }

        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
        // 找到所有点中的地面点或者距离为FLT_MAX(rangeMat的初始值)的点，并将他们标记为-1
		// rangeMat[i][j]==FLT_MAX，代表的含义是什么？ 无效点
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){ // 判断为地面的点不进行分割（所以labelMat.at<int>(i,j) = -1）
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }
        // 如果有节点订阅groundCloud，先把ground[ 8 * 1800]点保存到groundCloud中，后面会有发布操作
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }

    void cloudSegmentation(){
        // segmentation process
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                // 如果labelMat[i][j]=0,表示没有对该点进行过分类
				// 需要对该点进行聚类
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);

        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {

            // segMsg.startRingIndex[i]
			// segMsg.endRingIndex[i]
			// 表示第i线的点云起始序列和终止序列
			// 去除每束激光前5个点
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    // outliers that will not be used for optimization (always continue)
                    // labelMat数值为999999表示这个点是因为聚类数量不够30而被舍弃的点
					// 需要舍弃的点直接continue跳过本次循环，
					// 当列数为5的倍数，并且行数较大，可以认为非地面点的，将它保存进异常点云(界外点云)中
					// 然后再跳过本次循环
                    if (labelMat.at<int>(i,j) == 999999){
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }
                    // // 如果是地面点,对于列数不为5的倍数的，直接跳过不处理(地面点云稀疏化) majority of ground points are skipped
                    if (groundMat.at<int8_t>(i,j) == 1){
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }
                    // 上面多个if语句已经去掉了不符合条件的点，这部分直接进行信息的拷贝和保存操作
					// 保存完毕后sizeOfSegCloud递增

                    // mark ground points so they will not be considered as edge features later
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1); // 标记地面点
                    // mark the points' column index for marking occlusion later
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info, range信息为什么不保存在segmentedCloud->points[j + i*Horizon_SCAN].intensity中呢？？？？？？
                    // 因为fullCloud中的intensity = (float)rowIdn + (float)columnIdn / 10000.0，pushback给了segmentedCloud,后面featureAssociation中会用到
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of seg cloud
                    ++sizeOfSegCloud;
                }
            }

            // 以结束线前的第5线为结束
            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        
        // extract segmented cloud for visualization
        // 如果有节点订阅SegmentedCloudPure,
		// 那么把点云数据保存到segmentedCloudPure中去
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

    void labelComponents(int row, int col){
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        // 标准的BFS
        // BFS的作用是以(row，col)为中心向外面扩散，
        // 判断(row,col)是否是这个平面中一点
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;

            // neighbor=[[-1,0];[0,1];[0,-1];[1,0]]
			// 遍历点[fromIndX,fromIndY]边上的四个邻点      Loop through all the neighboring grids of popped grid
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // 是个环状的图片，左右连通     at range image margin (left or right side)
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;

                // prevent infinite loop (caused by put already examined point back)
                // 如果点[thisIndX,thisIndY]已经标记过,continue
				// labelMat中，-1代表无效点，0代表未进行标记过，其余为其他的标记
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

                // alpha代表角度分辨率，
				// X方向上 角度分辨率是segmentAlphaX(rad)
				// Y方向上 角度分辨率是segmentAlphaY(rad)
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                // 通过下面的公式计算这两点之间是否有平面特征
				// atan2(y,x)的值越大，d1，d2之间的差距越小,越平坦
                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta){  // 这里是一个聚类过程,大于60度并不能说明是一个平面

                    // segmentTheta=1.0472<==>60度
					// 如果算出角度大于60度，则假设这是个平面
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // 如果聚类超过30个点，直接标记为一个可用聚类，labelCount需要递增       check if this segment is valid
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        // 如果聚类点数小于30大于等于5，统计竖直方向上的聚类点数
        else if (allPushedIndSize >= segmentValidPointNum){
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            // 竖直方向上超过3个也将它标记为有效聚类
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;            
        }
        // segment is valid, mark these points
        if (feasibleSegment == true){
            ++labelCount;
        }else{ // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    // 发布各类点云内容
    void publishCloud(){
        // 1. Publish Seg Cloud Info
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);      // cloud_info 自定义消息类型，featureAssociation中有订阅此话题："/segmented_cloud_info"
        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;

        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);    // featureAssociation中有订阅此话题："/outlier_cloud"
        // segmented cloud with ground
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);  // featureAssociation中有订阅此话题："/segmented_cloud"
        // projected full cloud
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }
        // original dense ground cloud
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
