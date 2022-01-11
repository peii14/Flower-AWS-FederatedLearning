#!/bin/sh
for i in `seq 0 1`; do
    echo "Starting client $i"
    python client1.py --partition=${i} &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait