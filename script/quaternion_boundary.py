import pandas as pd
import matplotlib.pyplot as plt

import math

def quaternion_to_euler(x, y, z, w):

    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    
    return roll, pitch, yaw


# Read the CSV file
df = pd.read_csv('/home/joshua/Downloads/CPC16_Z1 (1).csv')

# column_w = df["q_w"]
# column_x = df["q_x"]
# column_y = df["q_y"]
# column_z = df["q_z"]
column = df[["q_w", "q_x", "q_y", "q_z"]]

euler_angles_x = []
euler_angles_y = []
euler_angles_z = []

for index, row in column.iterrows():
    roll, pitch, yaw = quaternion_to_euler(row["q_x"], row["q_y"], row["q_z"], row["q_w"])
    euler_angles_x.append(roll)
    euler_angles_y.append(pitch)
    euler_angles_z.append(yaw)

print(f"min roll: {min(euler_angles_x)}, max roll: {max(euler_angles_x)}")
print(f"min pitch: {min(euler_angles_y)}, max pitch: {max(euler_angles_y)}")
print(f"min yaw: {min(euler_angles_z)}, max yaw: {max(euler_angles_z)}")


plt.plot(euler_angles_x, label="x")
plt.plot(euler_angles_y, label="y")
plt.plot(euler_angles_z, label="z")
plt.legend()
plt.show()


