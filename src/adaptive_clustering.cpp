// Copyright (C) 2018  Zhi Yan and Li Sun

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include "adaptive_clustering/ClusterArray.h"

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

// Bounding Boxes
#include <jsk_recognition_msgs/BoundingBoxArray.h>

//#define LOG

ros::Publisher cluster_array_pub_;
ros::Publisher cloud_filtered_pub_;
ros::Publisher pose_array_pub_;
ros::Publisher bboxes_pub_;

bool print_fps_;
float z_axis_min_;
float z_axis_max_;
int cluster_size_min_;
int cluster_size_max_;

const int region_max_ = 10;  // Change this value to match how far you want to detect.
int regions_[100];

int frames;
clock_t start_time;
bool reset = true;  // fps
void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2_in)
{
  if (print_fps_)
    if (reset)
    {
      frames = 0;
      start_time = clock();
      reset = false;
    }  // fps

  /*** Convert ROS message to PCL ***/
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);

  /*** Remove ground and ceiling ***/
  pcl::IndicesPtr pc_indices(new std::vector<int>);
  pcl::PassThrough<pcl::PointXYZI> pt;
  pt.setInputCloud(pcl_pc_in);
  pt.setFilterFieldName("z");
  pt.setFilterLimits(z_axis_min_, z_axis_max_);
  pt.filter(*pc_indices);

  /*** Divide the point cloud into nested circular regions ***/
  boost::array<std::vector<int>, region_max_> indices_array;
  for (int i = 0; i < pc_indices->size(); i++)
  {
    float range = 0.0;
    for (int j = 0; j < region_max_; j++)
    {
      float d2 = pcl_pc_in->points[(*pc_indices)[i]].x * pcl_pc_in->points[(*pc_indices)[i]].x +
                 pcl_pc_in->points[(*pc_indices)[i]].y * pcl_pc_in->points[(*pc_indices)[i]].y +
                 pcl_pc_in->points[(*pc_indices)[i]].z *
                     pcl_pc_in->points[(*pc_indices)[i]].z;  // TODO move this to outer loop?

      if (d2 > range * range && d2 <= (range + regions_[j]) * (range + regions_[j]))
      {
        indices_array[j].push_back((*pc_indices)[i]);
        break;
      }
      range += regions_[j];
    }
  }

  /*** Euclidean clustering ***/
  float tolerance = 0.0;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZI>::Ptr> >
      clusters;

  for (int i = 0; i < region_max_; i++)
  {
    tolerance += 0.1;
    if (indices_array[i].size() > cluster_size_min_)
    {
      boost::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
      pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
      tree->setInputCloud(pcl_pc_in, indices_array_ptr);

      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
      ec.setClusterTolerance(tolerance);
      ec.setMinClusterSize(cluster_size_min_);
      ec.setMaxClusterSize(cluster_size_max_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(pcl_pc_in);
      ec.setIndices(indices_array_ptr);
      ec.extract(cluster_indices);

      for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end();
           it++)
      {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        {
          cluster->points.push_back(pcl_pc_in->points[*pit]);
        }
        cluster->width = cluster->size();
        cluster->height = 1;
        cluster->is_dense = true;
        clusters.push_back(cluster);
      }
    }
  }

  /*** Output ***/
  if (cloud_filtered_pub_.getNumSubscribers() > 0)
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_out(new pcl::PointCloud<pcl::PointXYZI>);
    sensor_msgs::PointCloud2 ros_pc2_out;
    pcl::copyPointCloud(*pcl_pc_in, *pc_indices, *pcl_pc_out);
    pcl::toROSMsg(*pcl_pc_out, ros_pc2_out);
    cloud_filtered_pub_.publish(ros_pc2_out);
  }

  adaptive_clustering::ClusterArray cluster_array;
  geometry_msgs::PoseArray pose_array;
  jsk_recognition_msgs::BoundingBoxArray bboxes;

  for (int i = 0; i < clusters.size(); i++)
  {
    if (cluster_array_pub_.getNumSubscribers() > 0)
    {
      sensor_msgs::PointCloud2 ros_pc2_out;
      pcl::toROSMsg(*clusters[i], ros_pc2_out);
      cluster_array.clusters.push_back(ros_pc2_out);
    }

    if (pose_array_pub_.getNumSubscribers() > 0)
    {
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*clusters[i], centroid);

      geometry_msgs::Pose pose;
      pose.position.x = centroid[0];
      pose.position.y = centroid[1];
      pose.position.z = centroid[2];
      pose.orientation.w = 1;
      pose_array.poses.push_back(pose);

#ifdef LOG
      Eigen::Vector4f min, max;
      pcl::getMinMax3D(*clusters[i], min, max);
      std::cerr << ros_pc2_in->header.seq << " " << ros_pc2_in->header.stamp << " " << min[0] << " " << min[1] << " "
                << min[2] << " " << max[0] << " " << max[1] << " " << max[2] << " " << std::endl;
#endif
    }

    if (bboxes_pub_.getNumSubscribers() > 0)
    {
      // Minimal Bounding Box
      pcl::PointXYZI origMinPoint, origMaxPoint;
      pcl::getMinMax3D(*clusters[i], origMinPoint, origMaxPoint);

      // Set z components zero
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud_transformed(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::copyPointCloud(*clusters[i], *cluster_cloud_transformed);
      for (size_t idx = 0u; idx < cluster_cloud_transformed->size(); ++idx)
        cluster_cloud_transformed->points[idx].z = 0;

      // Compute principal directions
      Eigen::Vector4f pcaCentroid;
      pcl::compute3DCentroid(*cluster_cloud_transformed, pcaCentroid);
      Eigen::Matrix3f covariance;
      computeCovarianceMatrixNormalized(*cluster_cloud_transformed, pcaCentroid, covariance);
      Eigen::Matrix2f smallcovariance = covariance.block(0, 0, 2, 2);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigen_solver(smallcovariance, Eigen::ComputeEigenvectors);
      Eigen::Matrix2f eigenVectorsPCA = eigen_solver.eigenvectors();

      // Transform the original cloud to the origin where the principal components correspond to the axes.
      Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
      projectionTransform.block<2, 2>(0, 0) = eigenVectorsPCA.transpose();
      projectionTransform.block<2, 1>(0, 3) = -1.f * (projectionTransform.block<2, 2>(0, 0) * pcaCentroid.head<2>());
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPointsProjected(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::transformPointCloud(*cluster_cloud_transformed, *cloudPointsProjected, projectionTransform);

      // Get the minimum and maximum points of the transformed cloud.
      pcl::PointXYZI minPoint, maxPoint;
      pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
      const Eigen::Vector2f meanXY = 0.5f * (maxPoint.getVector3fMap().head<2>() + minPoint.getVector3fMap().head<2>());
      const float meanZ = 0.5f * (origMaxPoint.z + origMinPoint.z);

      Eigen::Vector2f bboxTransform = eigenVectorsPCA * meanXY + pcaCentroid.head<2>();
      Eigen::Vector3f position(bboxTransform[0], bboxTransform[1], meanZ);
      float orientation = std::atan2(eigenVectorsPCA.col(0)[1], eigenVectorsPCA.col(0)[0]);
      Eigen::Vector3f dimension(maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, origMaxPoint.z - origMinPoint.z);

      jsk_recognition_msgs::BoundingBox bbox;
      bbox.pose.position.x = bboxTransform[0];
      bbox.pose.position.y = bboxTransform[1];
      bbox.pose.position.z = meanZ;
      bbox.pose.orientation.x = 0;
      bbox.pose.orientation.y = 0;
      bbox.pose.orientation.z = std::sin(orientation / 2.0);
      bbox.pose.orientation.w = std::cos(orientation / 2.0);
      bbox.dimensions.x = dimension[0];
      bbox.dimensions.y = dimension[1];
      bbox.dimensions.z = dimension[2];
      bbox.header.frame_id = ros_pc2_in->header.frame_id;
      bbox.header.stamp = ros_pc2_in->header.stamp;
      bbox.value = clusters[i]->size();

      bboxes.boxes.push_back(bbox);
    }

    bboxes.header.frame_id = ros_pc2_in->header.frame_id;
    bboxes_pub_.publish(bboxes);
  }

  if (cluster_array.clusters.size())
  {
    cluster_array.header = ros_pc2_in->header;
    cluster_array_pub_.publish(cluster_array);
  }

  if (pose_array.poses.size())
  {
    pose_array.header = ros_pc2_in->header;
    pose_array_pub_.publish(pose_array);
  }

  if (print_fps_)
    if (++frames > 10)
    {
      std::cerr << "[adaptive_clustering] fps = " << float(frames) / (float(clock() - start_time) / CLOCKS_PER_SEC)
                << ", timestamp = " << clock() / CLOCKS_PER_SEC << std::endl;
      reset = true;
    }  // fps
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "adaptive_clustering");

  /*** Subscribers ***/
  ros::NodeHandle nh;
  ros::Subscriber point_cloud_sub =
      nh.subscribe<sensor_msgs::PointCloud2>("/autonomy_module_lidar/points", 1, pointCloudCallback);

  /*** Publishers ***/
  ros::NodeHandle private_nh("~");
  cluster_array_pub_ = private_nh.advertise<adaptive_clustering::ClusterArray>("clusters", 100);
  cloud_filtered_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 100);
  pose_array_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100);
  bboxes_pub_ = private_nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("bboxes", 100, true);

  /*** Parameters ***/
  std::string sensor_model;

  private_nh.param<std::string>("sensor_model", sensor_model, "HDL-64E");  // VLP-16, HDL-32E, HDL-64E
  private_nh.param<bool>("print_fps", print_fps_, false);
  private_nh.param<float>("z_axis_min", z_axis_min_, -0.8);
  private_nh.param<float>("z_axis_max", z_axis_max_, 2.0);
  private_nh.param<int>("cluster_size_min", cluster_size_min_, 3);
  private_nh.param<int>("cluster_size_max", cluster_size_max_, 2200000);

  // Divide the point cloud into nested circular regions centred at the sensor.
  // For more details, see our IROS-17 paper "Online learning for human classification in 3D LiDAR-based tracking"
  if (sensor_model.compare("VLP-16") == 0)
  {
    regions_[0] = 2;
    regions_[1] = 3;
    regions_[2] = 3;
    regions_[3] = 3;
    regions_[4] = 3;
    regions_[5] = 3;
    regions_[6] = 3;
    regions_[7] = 2;
    regions_[8] = 3;
    regions_[9] = 3;
    regions_[10] = 3;
    regions_[11] = 3;
    regions_[12] = 3;
    regions_[13] = 3;
  }
  else if (sensor_model.compare("HDL-32E") == 0)
  {
    regions_[0] = 4;
    regions_[1] = 5;
    regions_[2] = 4;
    regions_[3] = 5;
    regions_[4] = 4;
    regions_[5] = 5;
    regions_[6] = 5;
    regions_[7] = 4;
    regions_[8] = 5;
    regions_[9] = 4;
    regions_[10] = 5;
    regions_[11] = 5;
    regions_[12] = 4;
    regions_[13] = 5;
  }
  else if (sensor_model.compare("HDL-64E") == 0)
  {
    regions_[0] = 14;
    regions_[1] = 14;
    regions_[2] = 14;
    regions_[3] = 15;
    regions_[4] = 14;
  }
  else
  {
    ROS_FATAL("Unknown sensor model!");
  }

  ros::spin();

  return 0;
}
