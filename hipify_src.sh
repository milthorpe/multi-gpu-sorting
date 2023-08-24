#!/usr/bin/env bash
mkdir -p hipsrc
for file in `ls src/*.cuh`
do
	echo $file
	hipfile=hip${file}
	hipify-perl $file > $hipfile
done

for file in `ls src/*.cu`
do
	echo $file
	hipfile=hip${file}
	hipify-perl $file > $hipfile
done

