#!/bin/bash
for i in $(seq 0 600); do
    sbatch job.sh "$i"
done