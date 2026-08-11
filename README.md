# Scan-Based RRTX Path Planner for TurtleBot3

A ROS 2 implementation of a simplified **RRTX-style dynamic path planner** for TurtleBot3 using live LiDAR data.

Unlike the map-based version, this planner does not use `/map`. It builds and updates its own occupancy grid from `/scan` measurements while using `/odom` as the planning coordinate frame.

## Files

```text
rrtx_algorithm_scan.py
rrtx_node_scan.py
```

### `rrtx_algorithm_scan.py`

Implements:

* 2-D occupancy grid
* LiDAR ray clearing
* obstacle inflation
* collision checking
* RRTX graph construction
* `g` and `lmc` cost updates
* graph rewiring and repair
* blocked-edge removal
* reconnection of newly free edges
* path extraction and smoothing
* graph pruning

### `rrtx_node_scan.py`

Handles the ROS 2 interface:

* subscribes to `/odom`
* subscribes to `/scan`
* transforms LiDAR points using TF2
* updates the occupancy grid
* runs the RRTX planner continuously
* publishes `/rrtx_path`
* publishes `/rrtx_grid`
* follows the path using `/cmd_vel`

---

## How It Works

The planner creates a fixed internal grid covering:

```text
X: -15 m to +15 m
Y: -15 m to +15 m
Resolution: 0.05 m
```

The robot pose is obtained from `/odom`.

Each valid LiDAR measurement is projected into the `odom` frame. Cells between the robot and the measured obstacle are marked free, while the LiDAR endpoint is marked occupied.

```text
Robot ---- free ---- free ---- free ---- Obstacle
```

The occupied cells are then inflated to account for the TurtleBot3 footprint.

```text
Robot radius   = 0.105 m
Safety margin  = 0.080 m
Inflation      = 0.185 m
```

The scan-based occupancy update is implemented directly in the grid using ray clearing followed by endpoint occupation.

---

## RRTX Planner

The RRTX graph is rooted at the goal.

The goal node starts with:

```text
g   = 0
lmc = 0
```

while new nodes initially have infinite cost.

The planner grows toward random samples and periodically biases sampling toward the current robot position.

Main parameters:

```text
STEP_SIZE        = 0.50 m
NEIGHBOR_RADIUS  = 1.20 m
GOAL_BIAS        = 0.10
MAX_TREE_NODES   = 800
```

## The planner maintains parent, child and neighbor relationships and uses `g` and `lmc` values with a priority queue to repair the graph when costs change.

## Dynamic Replanning

Before every planning cycle, the existing graph is checked against the latest LiDAR-generated grid.

If an obstacle appears:

```text
Old path:

Start -------- A -------- B -------- Goal

New obstacle:

Start -------- A ---- X ---- B -------- Goal
```

the blocked graph edge is removed and affected nodes are repaired.

If an obstacle later disappears, the planner also checks nearby nodes and can reconnect edges that have become collision-free again.

This allows the existing RRTX graph to be reused rather than rebuilding the complete planner every cycle.

---

## Planning Cycle

The planner runs every:

```text
0.2 seconds
```

or approximately:

```text
5 Hz
```

with up to:

```text
100 RRTX iterations
```

per planning callback.

The basic pipeline is:

```text
/odom
   |
/scan
   |
   v
Update occupancy grid
   |
   v
Inflate obstacles
   |
   v
Repair RRTX graph
   |
   v
Grow graph
   |
   v
Extract and smooth path
   |
   v
/rrtx_path
   |
   v
Waypoint controller
   |
   v
/cmd_vel
```

---

## ROS 2 Topics

### Subscribers

| Topic   | Type                    | Purpose                        |
| ------- | ----------------------- | ------------------------------ |
| `/odom` | `nav_msgs/Odometry`     | Robot position and orientation |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR obstacle detection       |

### Publishers

| Topic        | Type                     | Purpose                |
| ------------ | ------------------------ | ---------------------- |
| `/cmd_vel`   | `geometry_msgs/Twist`    | Robot motion           |
| `/rrtx_path` | `nav_msgs/Path`          | Current planned path   |
| `/rrtx_grid` | `nav_msgs/OccupancyGrid` | Inflated planning grid |

These interfaces are created directly by the ROS node.

---

## Goal Parameters

The target position is set using:

```text
goal_x
goal_y
```

Default:

```text
goal_x = 2.0
goal_y = 2.0
```

The goal is interpreted in the `odom` frame.

Example, if the package console entry is configured as `rrtx_node_scan`:

```bash
ros2 run rrtx_planner rrtx_node_scan --ros-args \
  -p goal_x:=2.0 \
  -p goal_y:=2.0
```

The exact executable name depends on the package's `setup.py`.

---

## Path Following

The node includes a simple waypoint controller.

```text
Goal tolerance       = 0.15 m
Waypoint tolerance   = 0.20 m

Linear command limit  = 0.15 m/s
Angular command limit = 0.50 rad/s
```

If the heading error is greater than `0.5 rad`, the robot rotates in place before moving forward.

The controller and its limits are implemented directly in `rrtx_node_scan.py`.

---

## RViz

Use:

```text
Fixed Frame: odom
```

Add:

```text
Path:
    /rrtx_path

Map:
    /rrtx_grid

LaserScan:
    /scan
```

Both the generated path and planner grid are published in the `odom` frame.

---

## Key Difference from the Map-Based Version

```text
Map-Based Version
/map + /scan
      |
      v
Static map + temporary obstacles
```

```text
Scan-Based Version
/odom + /scan
      |
      v
Occupancy grid built directly from LiDAR
```

The scan-based version can therefore operate without a saved map, but its environment representation depends on odometry accuracy and currently observed LiDAR data.

---

## Limitations

* No `/map` input
* planning uses the `odom` frame
* odometry drift can affect accumulated obstacle positions
* fixed `30 m x 30 m` planning area
* obstacles do not use a time-based expiration mechanism
* old obstacle cells are cleared only when later LiDAR rays pass through them
* simple waypoint controller
* no robot dynamics in the RRTX state
* neighbor searches do not use a spatial index such as a k-d tree
* simplified RRTX-style implementation rather than the complete formal RRTX algorithm

---

## Summary

This version combines:

```text
Odometry
   +
Live LiDAR
   +
Occupancy Grid
   +
Reusable RRTX Graph
   +
Dynamic Graph Repair
   +
Path Smoothing
   +
Waypoint Control
```

to provide real-time obstacle-aware navigation for TurtleBot3 without requiring a pre-built map.
