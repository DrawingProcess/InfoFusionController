# ADLAB-PLANNING

## Code Structure
```
ADLAB-PLANNING
|   main.py                                     # main: Multiple planning, control algorithms available
|   utils.py                                    # utils: calculate, tranform, ...
|
+---control
|       adaptive_mpc_controller.py              # control: adaptive mpc controller
|       base_controller.py                      # control: base controller
|       mi_mpc_purpursuit_controller.py         # control: mutual information mpc pure pursuit controller
|       mpc_controller.py                       # control: mpc controller
|       mpc_parallel_controller.py              # control: mpc parallel controller
|       multi_purpose_mpc_controller.py         # control: multi purpose mpc controller
|       pure_pursuit_controller.py              # control: pure pursuit controller
|       stanley_controller.py                   # control: stanley controller
|
+---map
|       complex_grid_map.py                     # map: random obstacle grid map
|       fixed_grid_map.py                       # map: fixed obstacle grid map
|       grid_map.py                             # map: base grid map
|       parking_lot.py                          # map: parking lot grid map
|
+---results
|       map_complex_grid_map.png                # map: random obstacle grid map
|       map_fixed_grid_map.png                  # map: fixed obstacle grid map
|       map_grid_map.png                        # map: base grid map
|       map_parking_lot.png                     # map: parking lot grid map
|
+---route_planner
|       a_star_route_planner.py                 # route planner: a star 
|       geometry.py                             # route planner: Pose, Node Class
|       hybrid_a_star_route_planner.py          # route planner: hybrid a star 
|       informed_rrt_star_planner.py            # route planner: informed rrt star
|       informed_rrt_star_smooth_planner.py     # route planner: informed rrt star (smoothing approach)
|       informed_trrt_star_planner.py           # route planner: informed trrt star
|       rrt_star_planner.py                     # route planner: rrt star
|       theta_star_planner.py                   # route planner: theta star 
|
\---test
        algotithm_speed_test.py                 # test: Comparison of algorithmic speed performance
```

## Env Settings

```shell
export QT_QPA_PLATFORM=xcb
export PYTHONPATH="$PYTHONPATH:$PWD"
```