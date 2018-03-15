#!/bin/bash 
path_to_data=../data/train
per_class=150

cd $path_to_data
classes=$(ls)

for class in *
do 
   class_dir="../validation/$class"
   mkdir $class_dir
   images=$(shuf -n$per_class -e $class/*.jpg )
   for img in $images
   do
       mv $img $class_dir
   done
#| xargs -0 -i echo {}
done

