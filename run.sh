#!/bin/bash

clear && g++ radius.cpp -lm -o RADIUS && ./RADIUS |tee out.log
