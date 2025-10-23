#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# put under /motion_primitive_library/test/ folder

"""
run_all_scenarios.py

- Generates a dataset of (A, B, obstacle) scenario .npz files and a CSV summary (same as your generator).
- For each scenario:
    - Builds a simple occupancy grid marking the circular obstacle as occupied.
    - Attempts to run map_planner_py's OccMapPlanner (if available) using planner logic you provided.
    - Visualizes vector field (from A), obstacle, start/goal, and planner trajectory (if planner ran).
    - Saves per-scenario PNG results.

Usage:
    python run_all_scenarios.py
"""
import os
import csv
import time
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import faulthandler; faulthandler.enable()

# ---------------------------
# Config (edit these if needed)
# ---------------------------
SAVE_DIR = "dataset_obstacle"
CSV_NAME = "dataset_summary.csv"
GRID_RESOLUTION = 0.2   # meters per cell for occupancy grid used by planner
GRID_EXTENT = 10.0      # half-extent, grid covers [-GRID_EXTENT, GRID_EXTENT] in x and y
START_POS = np.array([-8.0, -8.0], dtype=np.float32)
GOAL_POS = np.array([8.0, 8.0], dtype=np.float32)
# planner parameters
AMAX = 2.0
VMAX = 1.0
DT = 0.1

# ---------------------------
# Try to import map_planner_py
# ---------------------------
HAS_MPL = True
try:
    import map_planner_py as mpl
    print("Successfully imported map_planner_py:", mpl)
except Exception as e:
    HAS_MPL = False
    print("map_planner_py not available. Planner runs will be skipped.")
    print("Import error:", e)

# ---------------------------
# Dataset generation (same as your function)
# ---------------------------
def generate_dataset(save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    # Base dynamics
    A_base = np.array([[-1, -2], [2, -1]], dtype=np.float32)
    B_base = np.array([[0], [1]], dtype=np.float32)

    # Generate small perturbations
    dynamics_list = []
    rng = np.random.default_rng(1234)
    for i in range(5):
        A_var = A_base + 0.2 * rng.standard_normal((2, 2)).astype(np.float32)
        B_var = B_base + 0.1 * rng.standard_normal((2, 1)).astype(np.float32)
        dynamics_list.append((A_var, B_var))

    # Obstacle positions (kept same as your list)
    obstacle_positions = [
        (3, 5), (5, 9), (3, 3), (-3, 3), (-6, 8),
        (3, -5), (-3, -3), (-3, -8), (3, -3), (-3, -3)
    ]
    radius = 0.7

    metadata = []

    for dyn_idx, (A, B) in enumerate(dynamics_list):
        for obs_idx, center in enumerate(obstacle_positions):
            scenario_name = f'scenario_{dyn_idx}_{obs_idx}.npz'
            filepath = os.path.join(save_dir, scenario_name)
            np.savez(filepath, A=A, B=B, obstacle_center=np.array(center, dtype=np.float32), radius=np.float32(radius))

            A_flat = A.flatten()
            B_flat = B.flatten()
            metadata.append({
                'filename': scenario_name,
                'A11': float(A_flat[0]), 'A12': float(A_flat[1]), 'A21': float(A_flat[2]), 'A22': float(A_flat[3]),
                'B1': float(B_flat[0]), 'B2': float(B_flat[1]),
                'obstacle_center_x': float(center[0]),
                'obstacle_center_y': float(center[1]),
                'radius': float(radius)
            })
            print(f"Saved {scenario_name}")

    # Save CSV
    csv_path = os.path.join(save_dir, CSV_NAME)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata[0].keys()))
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\n✅ Dataset generated in '{save_dir}/' with {len(metadata)} scenarios.")
    print(f"📄 Summary CSV saved to '{csv_path}'")
    return metadata

# ---------------------------
# Helper: build occupancy grid from a single circular obstacle
# Returns origin (x,y), dims (nx, ny), flat data (0/1), resolution
# ---------------------------
def build_occ_grid_from_circle(center, radius, resolution=GRID_RESOLUTION, extent=GRID_EXTENT):
    # grid axis from -extent .. extent
    x_min, x_max = -extent, extent
    y_min, y_max = -extent, extent
    xs = np.arange(x_min, x_max + 1e-9, resolution)
    ys = np.arange(y_min, y_max + 1e-9, resolution)
    nx = xs.size
    ny = ys.size

    # create occupancy: 1 for occupied, 0 for free
    # map_data will be flattened in row-major order (x-fast or y-fast?) -- the example expects a 1D array.
    # We will follow the convention: iterate over y then x, so data order matches typical image flattening: row-major (y rows).
    occ = np.zeros((ny, nx), dtype=np.int8)
    cx, cy = center
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                occ[iy, ix] = 1

    # origin corresponds to lower-left corner (x_min, y_min)
    origin = np.array([x_min, y_min], dtype=np.float32)
    dims = np.array([nx, ny], dtype=np.int32)
    flat = occ.ravel().astype(np.int8)  # row-major flatten
    return origin, dims, flat, float(resolution)

# ---------------------------
# Planner runner using your MPL logic
# Accepts path to .npz scenario, start, goal. Returns dict with success info and saved image path.
# ---------------------------
def run_planner_on_scenario(npz_path, start_pos=START_POS, goal_pos=GOAL_POS, out_dir=SAVE_DIR):
    scenario_name = os.path.splitext(os.path.basename(npz_path))[0]
    out_img = os.path.join(out_dir, f"{scenario_name}_result.png")

    # load scenario
    data = np.load(npz_path, allow_pickle=True)
    A = data['A']
    B = data['B']
    center = tuple(np.array(data['obstacle_center']).astype(float))
    radius = float(data['radius'])

    print(f"\n=== Scenario: {scenario_name} ===")
    print("A =", A)
    print("B =", B)
    print("obstacle center, radius:", center, radius)

    # Build occupancy grid
    origin, dims, flat_map, res = build_occ_grid_from_circle(center, radius, resolution=GRID_RESOLUTION, extent=GRID_EXTENT)
    print("Built occupancy grid: origin", origin, "dims", dims, "res", res, "occupied cells", int(flat_map.sum()))

    planner_success = False
    traj = None
    close_set = np.empty((0, 2))

    if HAS_MPL:
        # Create map util and planner following your example
        try:
            map_util = mpl.OccMapUtil()
            # setMap expects: origin (2,), dim (2,), data (flat list/array), resolution (float)
            # convert flat_map to a python list or numpy array as required
            map_util.setMap(origin.astype(np.float32), dims.astype(np.int32), flat_map.astype(np.int8), float(res))
            map_util.freeUnknown()
            # Build Waypoint2D for start and goal
            start = mpl.Waypoint2D()
            start.pos = np.array(start_pos, dtype=np.float32)
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

            # control inputs U: sample simple accelerations in 2D (as your snippet)
            U = []
            du = AMAX
            for ax in np.arange(-AMAX, AMAX + 1e-9, du):
                for ay in np.arange(-AMAX, AMAX + 1e-9, du):
                    U.append(np.array([ax, ay], dtype=np.float32))

            # initialize planner (try both class names used in your example)
            try:
                planner = mpl.OccMapPlanner(True)
                planner.setMapUtil(map_util)
                planner.setVmax(VMAX)
                planner.setAmax(AMAX)
                planner.setDt(DT)
                planner.setU(U)
            except Exception:
                # fallback to MapPlanner2D alias
                planner = mpl.MapPlanner2D(True)
                planner.set_map_util(map_util)
                planner.set_vmax(VMAX)
                planner.set_amax(AMAX)
                planner.set_dt(DT)
                planner.set_u(U)

            # run planner
            t0 = time.time()
            valid = planner.plan(start, goal)
            t_elapsed = (time.time() - t0) * 1000.0
            print(f"Planner run: valid={valid} took {t_elapsed:.2f} ms")
            close_set = np.array(planner.getCloseSet()) if hasattr(planner, "getCloseSet") else np.empty((0,2))
            if valid:
                traj = np.array(planner.getTraj())
                planner_success = True
            else:
                planner_success = False

        except Exception as e:
            print("Planner or map_util raised exception; skipping planner for this scenario.")
            print("Exception:", e)
            HAS_LOCAL_MPL = False
    else:
        print("Skipping planner run because map_planner_py is not available in this environment.")

    # ---------------------------
    # Visualization (vector field, obstacle, start/goal, close set, trajectory)
    # ---------------------------
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')

    # Vector field from A
    x = np.linspace(-GRID_EXTENT, GRID_EXTENT, 20)
    y = np.linspace(-GRID_EXTENT, GRID_EXTENT, 20)
    X, Y = np.meshgrid(x, y)
    Uv = A[0,0]*X + A[0,1]*Y
    Vv = A[1,0]*X + A[1,1]*Y
    ax.quiver(X, Y, Uv, Vv, alpha=0.6)

    # obstacle
    circ = plt.Circle(center, radius, color='red', fill=False, linestyle='--', linewidth=1.5)
    ax.add_patch(circ)
    ax.scatter(center[0], center[1], color='red', s=50, label='obstacle center')

    # start / goal
    ax.scatter(start_pos[0], start_pos[1], c='green', s=60, label='start')
    ax.scatter(goal_pos[0], goal_pos[1], c='cyan', s=60, label='goal')

    # close set
    if close_set.size > 0:
        try:
            ax.scatter(close_set[:,0], close_set[:,1], c='gray', s=8, label='expanded')
        except Exception:
            pass

    # trajectory
    if traj is not None and traj.size > 0:
        try:
            ax.plot(traj[:,0], traj[:,1], c='blue', linewidth=2.0, label='trajectory')
        except Exception:
            pass

    ax.set_xlim(-GRID_EXTENT, GRID_EXTENT)
    ax.set_ylim(-GRID_EXTENT, GRID_EXTENT)
    ax.grid(True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"{scenario_name} planner_success={planner_success}")
    ax.legend(loc='upper right')

    # Save figure
    fig.tight_layout()
    plt.savefig(out_img, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_img}")

    return dict(
        scenario=scenario_name, planner_success=planner_success, traj_exists=(traj is not None),
        img_path=out_img, close_set_count=(close_set.shape[0] if close_set is not None else 0)
    )

# ---------------------------
# Main
# ---------------------------
def main():
    # 1) Generate dataset (if not already present)
    metadata = None
    csv_path = os.path.join(SAVE_DIR, CSV_NAME)
    if not os.path.exists(SAVE_DIR) or not os.path.exists(csv_path):
        metadata = generate_dataset(SAVE_DIR)
    else:
        # load CSV to get metadata list
        import csv as _csv
        metadata = []
        with open(csv_path, newline='') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                metadata.append(row)
        print(f"Loaded existing dataset metadata with {len(metadata)} entries.")

    # 2) Loop through scenarios and run planner + visualization
    results = []
    for entry in metadata:
        scenario_file = os.path.join(SAVE_DIR, entry['filename'])
        if not os.path.exists(scenario_file):
            print("Skipping missing file:", scenario_file)
            continue
        res = run_planner_on_scenario(scenario_file, start_pos=START_POS, goal_pos=GOAL_POS, out_dir=SAVE_DIR)
        results.append(res)

    # Summary
    succ = sum(1 for r in results if r['planner_success'])
    print(f"\n=== Batch finished: {len(results)} scenarios processed, {succ} succeeded ===")
    print("Per-scenario outputs (first 10):")
    for r in results[:10]:
        print(r)

if __name__ == "__main__":
    main()
