#! /bin/bash

cd $1;

for scene in `ls samples/`; do
    for channel in `ls samples/$scene/`; do
        for frame in `ls samples/$scene/$channel/`; do
            rm -f samples/$scene/$channel/$frame &
        done
        wait
    done
done

for scene in `ls sweeps/`; do
    for channel in `ls sweeps/$scene/`; do
        for frame in `ls sweeps/$scene/$channel/`; do
            rm -f sweeps/$scene/$channel/$frame &
        done
        wait
    done
done

cd .. && rm -rf $1