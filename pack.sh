#!/bin/bash

# Prepare tmp folder
DIR=pv021_project
if [ -e $DIR ]; then
    echo "$DIR exists! Remove it before continuing"
fi
mkdir $DIR


# src/ folder
cp -r src/ $DIR/src/

# data/ folder
mkdir $DIR/data/

# Project assignment as the new README
cp assignment.md $DIR/README.md

# Rest of the files
cp Makefile run.sh pack.sh $DIR/


# Zip it
zip -r xhrabos_xskalos.zip $DIR/*
rm -rf $DIR
