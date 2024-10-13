# ADLAB-PLANNING

## Code Structure
```
ADLAB-PLANNING
|   main.py                                     # main: Multiple planning, control algorithms available
|   utils.py                                    # utils: calculate, tranform, ...
│   test.sh                                     # test: test shell scripts
|
├── conf
│       map_custom.json                         # conf: complex rectangle road
│       map_easy.json                           # conf: Re-imple map_easy in informed TRRT Star paper 
│       map_hard.json                           # conf: Re-imple map_hard in informed TRRT Star paper 
│       map_medium.json                         # conf: Re-imple map_medium in informed TRRT Star paper 
│
├---control
|       adaptive_mpc_controller.py              # control: adaptive mpc controller
|       base_controller.py                      # control: base controller
|       info_fusion_controller.py                 # control: pure pursuit + mpc controller (mutual information)
|       mpc_mi_controller.py                    # control: combinate multiple mpc controller (mutual information)
|       mpc_controller.py                       # control: mpc controller
|       mpc_parallel_controller.py              # control: mpc parallel controller
|       multi_purpose_mpc_controller.py         # control: multi purpose mpc controller
|       pure_pursuit_controller.py              # control: pure pursuit controller
|       stanley_controller.py                   # control: stanley controller
|
├---map
|       random_grid_map.py                      # map: random obstacle grid map
|       fixed_grid_map.py                       # map: fixed obstacle grid map
|       grid_map.py                             # map: base grid map
|       parking_lot.py                          # map: parking lot grid map
|
├---results
|       map_random_grid_map.png                 # results: random obstacle grid map
|       map_fixed_grid_map.png                  # results: fixed obstacle grid map
|       map_grid_map.png                        # results: base grid map
|       map_parking_lot.png                     # results: parking lot grid map
|
├---route_planner
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
        test_controller.py                      # test: controller
        test_route_planner.py                   # test: route planner
```

## Env Settings

```shell
conda env create -f planning.yaml
conda activate planning
```

```shell
export QT_QPA_PLATFORM=xcb
export PYTHONPATH="$PYTHONPATH:$PWD"
```