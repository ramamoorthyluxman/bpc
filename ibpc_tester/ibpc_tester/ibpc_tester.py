# todo(Yadunund): Add copyright.

import rclpy
from rclpy.node import Node

# from bop_toolkit_lib import *

from ibpc_interfaces.msg import PoseEstimate
from ibpc_interfaces.srv import GetPoseEstimates

import sys

class TesterNode(Node):
    def __init__(self):
        super().__init__('ibpc_tester_node')
        self.get_logger().info('Tester node started...')


def main(argv=sys.argv):
    rclpy.init(args=argv)
    args_without_ros = rclpy.utilities.remove_ros_args(argv)
    n = TesterNode()
    try:
        rclpy.spin(n)
        rclpy.get_logger.info('Shutting down...')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()