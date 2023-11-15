#!/bin/bash

# Update the package list
sudo apt-get update

# Read each line in the packages.txt file and install the package
while IFS= read -r package; do
    sudo apt-get install -y "$package"
done < "packages.txt"
