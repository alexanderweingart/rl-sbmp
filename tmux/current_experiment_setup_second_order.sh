#!/bin/bash

tmux new-session -d -s bugtrap_second \; \
  split-window -v \; \
  split-window -v \; \
  split-window -v \; \
  select-layout even-vertical \; \
  send-keys -t 1 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 2 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 3 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 1 'source venv/bin/activate' C-m \; \
  send-keys -t 1 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/second-order-car/bugtrap/2023-07-25_6_second_order_car_bugtrap_rl.yaml' C-m \; \
  send-keys -t 2 'source venv/bin/activate' C-m \; \
  send-keys -t 2 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/second-order-car/bugtrap/2023-07-25_7_second_order_car_bugtrap_mcp.yaml' C-m \; \
  send-keys -t 3 'source venv/bin/activate' C-m \; \
  send-keys -t 3 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/second-order-car/bugtrap/2023-07-25_8_second_order_car_bugtrap_mcp_guided.yaml' C-m \; 

tmux new-session -d -s zigzag_second \; \
  split-window -v \; \
  split-window -v \; \
  split-window -v \; \
  select-layout even-vertical \; \
  send-keys -t 1 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 2 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 3 'cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..' C-m \; \
  send-keys -t 1 'source venv/bin/activate' C-m \; \
  send-keys -t 1 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/second-order-car/zigzag/2023-07-25_10_second_order_car_zigzag_mcp.yaml' C-m \; \
  send-keys -t 2 'source venv/bin/activate' C-m \; \
  send-keys -t 2 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/second-order-car/zigzag/2023-07-25_11_second_order_car_zigzag_mcp_guided.yaml' C-m \; \
  send-keys -t 3 'source venv/bin/activate' C-m \; \
  send-keys -t 3 'python motion_planning/evaluator/motion_planning_evaluator.py ao-rrt --conf motion_planning/evaluator/experiment_configs/cluster/thesis/second-order-car/zigzag/2023-07-25_9_second_order_car_zigzag_rl.yaml' C-m \; 
