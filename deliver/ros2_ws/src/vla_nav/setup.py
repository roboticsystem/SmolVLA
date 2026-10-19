from setuptools import find_packages, setup
package_name = "vla_nav"
setup(name=package_name, version="0.1.0",
      packages=find_packages(exclude=["test"]),
      data_files=[("share/ament_index/resource_index/packages",
                   ["resource/" + package_name]),
                  ("share/" + package_name, ["package.xml"]),
                  ("share/" + package_name + "/launch", ["launch/nav.launch.py"]),
                  ("share/" + package_name + "/config", ["config/params.yaml"])],
      install_requires=["setuptools"],
      entry_points={"console_scripts": [
          "camera_node = vla_nav.camera_node:main",
          "lidar_node = vla_nav.lidar_node:main",
          "odom_node = vla_nav.odom_node:main",
          "control_node = vla_nav.control_node:main",
          "vla_bridge_node = vla_nav.vla_bridge_node:main"]})
