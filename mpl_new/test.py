import sys
import os
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import faulthandler; faulthandler.enable()

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_dir, "build"))

import map_planner_py as mpl

print(mpl)

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'


with open("../data/corridor.yaml") as f:
    yaml_list = yaml.safe_load(f)

data = {list(d.keys())[0]: list(d.values())[0] for d in yaml_list}

start_pos = data["start"]  # e.g., [x, y]
goal_pos = data["goal"]    # e.g., [x, y]
map_origin = np.array(data["origin"], dtype=np.float32)
map_dim = np.array(data["dim"], dtype=np.int32)
map_data = np.asarray(data["data"], dtype=np.int8).ravel()  #转 list，符合 pybind11 接口
map_res = float(data["resolution"])

print("Start:", start_pos)
print("Goal:", goal_pos)
print("Map origin:", map_origin)
print("Map dim:", map_dim)
print("Map res:", map_res)
print("Map data len:", len(map_data))

print("load map...\n\n")


map_util = mpl.OccMapUtil()
print("map_dim product:", map_dim[0]*map_dim[1], "len(map_data):", map_data.size if hasattr(map_data, "size") else len(map_data))
map_util.setMap(map_origin, map_dim, map_data, map_res)
map_util.freeUnknown()
print("Map set.")

# except Exception as e:
#     # print(e.stacktrace)
#     print("Error in setting map:", e)
#     exit(1)

print(f"{YELLOW} resolution: {RESET} {map_res}")

try:
    # start_pos_scaled = start_pos / map_res
    start = mpl.Waypoint2D()
    start.pos = start_pos
except Exception as e:
    print(f"exception: {e}")
    print(e.stacktrace)
print(f"{YELLOW}type(start_pos){RESET}: {type(start_pos)}, {YELLOW}start_pos{RESET}: {start_pos}")

start.vel = np.zeros(2, dtype=np.float32)
start.acc = np.zeros(2, dtype=np.float32)
start.jrk = np.zeros(2, dtype=np.float32)
start.use_pos = True
start.use_vel = True
start.use_acc = False
start.use_jrk = False
start.use_yaw = False

goal = mpl.Waypoint2D()
goal.pos = np.array(goal_pos, dtype=np.float32)
goal.vel = np.zeros(2, dtype=np.float32)
goal.acc = np.zeros(2, dtype=np.float32)
goal.jrk = np.zeros(2, dtype=np.float32)
goal.use_pos = True
goal.use_vel = False
goal.use_acc = False
goal.use_jrk = False
goal.use_yaw = False

print(f"{YELLOW}Waypoint 2d allset{RESET}")

# === 控制输入 ===
amax = 2.0
du = amax
U = []
for ax in np.arange(-amax, amax + 1e-9, du):
    for ay in np.arange(-amax, amax + 1e-9, du):
        U.append(np.array([ax, ay], dtype=np.float32))

print(f"{YELLOW}list U: {U} {RESET}")

# # === 初始化 Planner ===
# 原版lpanner
planner = mpl.MapPlanner2D(True)
planner.set_map_util(map_util)
planner.set_vmax(1.0)
planner.set_amax(amax)
planner.set_dt(0.1)
planner.set_u(U)
print(f"{YELLOW}MapPlanner2D done.{RESET}")

# alias
planner = mpl.OccMapPlanner(True)
planner.setMapUtil(map_util)
planner.setVmax(1.0)
planner.setAmax(amax)
planner.setDt(0.1)
planner.setU(U)
print(f"{YELLOW}OccMapPlanner done. {RESET}")

# # === 开始规划 ===
start_time = time.time()
valid = planner.plan(start, goal)
elapsed_ms = (time.time() - start_time) * 1000
print(f"{YELLOW}MPL Planner takes{RESET}: {elapsed_ms:.2f} ms")
print(f"{YELLOW}Expanded states{RESET}: {len(planner.getCloseSet())}")

# # === Matplotlib 绘图 ===
fig, ax = plt.subplots()
ax.set_aspect('equal')
# ax.set_xlim(0, map_dim[0])
# ax.set_ylim(0, map_dim[1])
# ax.invert_yaxis()
x_min = map_origin[0]
x_max = map_origin[0] + map_dim[0] * map_res
y_min = map_origin[1]
y_max = map_origin[1] + map_dim[1] * map_res

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# # 绘制障碍物
cloud = np.array(map_util.getCloud())
if cloud.size > 0:
    ax.scatter(cloud[:, 0], cloud[:, 1], c='black', s=5, label='obstacles')

# # 绘制搜索状态
close_set = np.array(planner.getCloseSet())
if close_set.size > 0:
    ax.scatter(close_set[:, 0], close_set[:, 1], c='gray', s=10, label='expanded')
print(f"{YELLOW}closed set:{RESET} {close_set}")


# # 绘制起点和终点
ax.scatter(start.pos[0], start.pos[1], c='red', s=50, label='start')
ax.scatter(goal.pos[0], goal.pos[1], c='cyan', s=50, label='goal')

# # 绘制轨迹
if valid:
    traj = np.array(planner.getTraj())
    ax.plot(traj[:, 0], traj[:, 1], c='blue', linewidth=2, label='trajectory')

ax.legend()
ax.set_title("MPL Planner Result")
output_path = os.path.join(os.path.dirname(__file__), "test_planner_2d_result_matplotlib.jpg")
plt.savefig(output_path)
print(f"Saved results to {output_path}")
plt.show()

print(f"{YELLOW}Saved results to test_planner_2d_result_matplotlib.jpg{RESET}")

