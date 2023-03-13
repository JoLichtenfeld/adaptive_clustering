import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
import std_msgs.msg
import numpy as np
from scipy.spatial.transform import Rotation as R


rospy.init_node("pcl_pub", anonymous=True)

print("Initializing")
pub = rospy.Publisher("/autonomy_module_lidar/points", PointCloud2, queue_size=1,)

rospy.sleep(0.5)

points1 = []
# !!! For some reason it does not work if cloud is quadratic in x and y direction!
for z in [0, 0.05]:
    for x in np.linspace(-1, 1, 51):
        for y in np.linspace(-1.5, 1.5, 51):
            points1.append(np.asarray([x, y, z, 1]))
print("Created %i points" % len(points1))

r = R.from_euler('z', 5, degrees=True)
T = np.zeros((4,4))
T[:3,:3] = r.as_matrix()
T[-1,-1] = 1

points_transf = [list(T@p) for p in points1]


##########################################################

points2 = []
for x in [1]: #, 1.05]:
    for z in np.linspace(0, 5, 101):
        for y in np.linspace(2, 6, 51):
            points2.append(np.asarray([x, y, z, 0]))
print("Created %i points" % len(points2))

r = R.from_euler('z', 45, degrees=True)
T = np.zeros((4,4))
T[:3,:3] = r.as_matrix()
T[-1,-1] = 1

points_transf += [list(T@p) for p in points2]


header = std_msgs.msg.Header()
header.frame_id = "autonomy_module_lidar_frame"
fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('intensity', 16, PointField.FLOAT32, 1)
          ]
cloud = pc2.create_cloud(header, fields, points_transf)

while not rospy.is_shutdown():

    cloud.header.stamp = rospy.Time.now()

    print("publishing...")
    pub.publish(cloud)
    rospy.sleep(1.0)

rospy.spin()
