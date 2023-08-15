#!/bin/bash

tmux new-session -d -s bugtrap_first \; \
  split-window -v \; \
  split-window -v \; \
  split-window -v \; \
  select-layout even-vertical \; \
  send-keys -t 1 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 2 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 3 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 1 'source venv/bin/activate' C-m \; \
  send-keys -t 1 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/first-order-car/fast/bugtrap/2023-07-29_0_first_order_car_fast_bugtrap_rl.yaml' C-m \; \
  send-keys -t 2 'source venv/bin/activate' C-m \; \
  send-keys -t 2 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/first-order-car/fast/bugtrap/2023-07-29_1_first_order_car_fast_bugtrap_mcp.yaml' C-m \; \
  send-keys -t 3 'source venv/bin/activate' C-m \; \
  send-keys -t 3 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/first-order-car/fast/bugtrap/2023-07-29_2_first_order_car_fast_bugtrap_mcp_guided.yaml' C-m \; 

tmux new-session -d -s zigzag_first \; \
  split-window -v \; \
  split-window -v \; \
  split-window -v \; \
  select-layout even-vertical \; \
  send-keys -t 1 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 2 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 3 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 1 'source venv/bin/activate' C-m \; \
  send-keys -t 1 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/first-order-car/fast/zigzag/2023-07-29_3_first_order_car_fast_zigzag_rl.yaml' C-m \; \
  send-keys -t 2 'source venv/bin/activate' C-m \; \
  send-keys -t 2 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/first-order-car/fast/zigzag/2023-07-29_4_first_order_car_fast_zigzag_mcp.yaml' C-m \; \
  send-keys -t 3 'source venv/bin/activate' C-m \; \
  send-keys -t 3 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/first-order-car/fast/zigzag/2023-07-29_5_first_order_car_fast_zigzag_mcp_guided.yaml' C-m \; 
