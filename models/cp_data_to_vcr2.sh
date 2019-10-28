#!/bin/bash

while true;do
	cp /data/vcr/rl_sampling-correct-3/saves/* /vcr2/vcr/rl_sampling-correct-3/saves/
	cp /data/tensorboard_log/vcr/rl_sampling-correct-3/* /vcr2/tensorboard_log/vcr/rl_sampling-correct-3/
	echo 'Done'
	sleep 600
done
