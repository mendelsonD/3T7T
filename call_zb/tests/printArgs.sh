#!/bin/bash

echo "[bash] Passed arguments:"

args="$@"

count=1
# print all arguments with index
for i in "$@"; do 
        echo -e "\t$count \t $i"
        ((count++))
        shift
done

exit 0

