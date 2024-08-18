# ADLAB-PLANNING

## Code Structure
```
ADLAB-PLANNING
|   algotithm_speed_test.py                     # Comparison of algorithmic speed performance
|   main.py                                     # main: Multiple planning, control algorithms available
|   utils.py                                    # utils: calculate, tranform, ...
|
+---control
|       mpc_adaptive.py                         # control: adaptive MPC
|       mpc_basic.py                            # control: basic MPC 
|       mpc_basic_parallel.py                   # control: basic MPC parallel processing
|       pure_pursuit.py                         # control: pure pursuit
|       stanley.py                              # control: stanley
|
+---results
|
+---route_planner
|       a_star_route_planner.py                 # route planner: a star 
|       geometry.py                             # Class Pose, Node
|       hybrid_a_star_route_planner.py          # route planner: hybrid a star 
|       informed_rrt_star_planner.py            # route planner: informed rrt star
|       informed_rrt_star_smooth_planner.py     # route planner: informed rrt star (smoothing approach)
|       informed_trrt_star_planner.py           # route planner: informed trrt star
|       rrt_star_planner.py                     # route planner: rrt star
|       theta_star_planner.py                   # route planner: theta star 
|
\---space
        grid_map.py                             # base grid map
        complex_grid_map.py                     # gridmap: random obstacle grid map
        parking_lot.py                          # gridmap: parking lot grid map
```