#!/bin/bash

# 循环执行程序 100 次
for i in {1..100}; do
  echo "执行程序，第 $i 次"
  seed=$(( RANDOM % (999999-100000+1) + 100000 ))

  timeout 600 python data_task_interface.py $seed
  if [ $? -eq 0 ]
  then
    echo "命令已完成"
  else
    echo "命令已超时，删除未完成的数据"
    rm -rf new_runs/seed-$seed
  fi
done