#!/bin/bash

echo "bash script with arguments"

args="$@"

count=1
# print all arguments with index
for i in "$@"; do 
        echo -e "$count \t $i"
        ((count++))
        shift
done

exit 0

