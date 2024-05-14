#!/bin/bash

Xvfb :1 -screen 0 1024x768x16 &

export DISPLAY=:1

sleep 3

roslaunch missions start_boat_channel.launch
