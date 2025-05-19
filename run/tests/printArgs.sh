#!/bin/bash
{
        nohup

        echo "[bash] Passed arguments:"

        args="$@"

        count=1
        # print all arguments with index
        for i in "$@"; do 
                echo -e "\t$count \t $i"
                ((count++))
                shift
        done
} > /host/verges/tank/data/daniel/out.log
        exit 0

