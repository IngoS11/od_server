# Watchdog to Detect Objects in Snapshots made by Synology Surveylance Station
This repository contains a set of python script that make use of Tensorflow Light to detect
objects in snapshots created by a Synology Surveylance Station. Surveylance station should
be configured to make multiple snapshots when motion is detected. As Surveylance Station creates
a lot of false positives through clouds and trees moving, Tensorflow Light will take
the images from the snapshot and identify the objects to reduce the number of false positives.
The scripts are intended to be run on a Raspberry PI4 which has the Snapshots folder mounted
from the Synology NAS.

## Prerequisites
The scripts are on a Raspberry PI4 4GB on [Raspberry PI OS Buster](https://downloads.raspberrypi.org/raspios_lite_armhf/images/raspios_lite_armhf-2021-01-12/2021-01-11-raspios-buster-armhf-lite.zip) in my house. Other combinations
might work but your milage may vary. 

## Installation
Install all Libraries and packages required.
```
sudo apt install python3-pip python3-pil python3-venv git libatlas-base-dev libopenjp2-7
```
Clone this repository and create a virtual python environment in it
```
git clone https://github.com/Ingos11/od_server
cd od_server
python3 -m venv .venv
```
Follow the Tensorflow Light Installation. At the time of writing the only thing that
was needed from there is.
```
pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl

```
Download and Install Models, Labels and Python Requirements for the project
```
download.sh
```

## Usage of the Object Watcher
tbd.