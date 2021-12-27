#!/bin/bash

for i in {0..8..1}; do
    echo ${i}
    grep -v home set${i}_seq.info | grep -v different  > ../valid_data.set/set${i}_seq.info
    
done
