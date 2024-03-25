#!/bin/bash
shopt -s nullglob
files=$(find . -type f -name "*.out")
shopt -u nullglob

if [ -z "$files" ]; then
    echo "No .out files found in the directory or its subdirectories."
    exit 1
fi

for filename in $files; do
    dirname=$(basename $(dirname "$filename"))
    echo "$dirname"
    result=$ grep -F  "FINAL SINGLE POINT ENERGY" $filename | tail -n1
done
