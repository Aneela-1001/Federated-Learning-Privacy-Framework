#!/bin/bash
# Bash script to start multiple DDFed clients

NUM_CLIENTS=${1:-3}
SERVER_ADDRESS=${2:-"127.0.0.1:8080"}
IS_BENIGN=${3:-"true"}

echo "Starting $NUM_CLIENTS clients..."

PROJECT_ROOT=$(dirname "$0")

for i in $(seq 1 $NUM_CLIENTS); do
    echo "Starting client-$i..."
    
    python "$PROJECT_ROOT/client/ddfed_client_main.py" \
        --server-address "$SERVER_ADDRESS" \
        --client-id $i \
        $([ "$IS_BENIGN" = "true" ] && echo "--is-benign") &
    
    sleep 1
done

echo "All $NUM_CLIENTS clients started!"
echo "Check the server terminal for training progress."
