#!/bin/bash


# This should be enough:
make -f Makefile run


# Or, legacy code:
# echo "#################"
# echo "    PREPARING    "
# echo "#################"
# 
# # module add gcc-10.2
# 
# echo "#################"
# echo "    COMPILING    "
# echo "#################"
# 
# make network
# 
# echo "#################"
# echo "     RUNNING     "
# echo "#################"
# 
# nice -n 19 ./network

