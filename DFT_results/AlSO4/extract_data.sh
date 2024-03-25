#!/bin/bash
shopt -s nullglob
files=$(find . -type f -name "*.out")
shopt -u nullglob

if [ -z "$files" ]; then
    echo "No .out files found in the directory or its subdirectories."
    exit 1
fi

for filename in $files; do
    result=$ grep -F  "FINAL SINGLE POINT ENERGY" $1 | tail -n1
done


# # Check if filename is provided as argument
# if [ $# -ne 1 ]; then
#     echo "Usage: $0 <filename>"
#     exit 1
# fi

# filename="$1"

# result=$ grep -F  "FINAL SINGLE POINT ENERGY" $1 | tail -n1

