# Getting Started
This document briefly describes how to install and use the code.

## Environment
We conducted experiments in the following environment:
 - Linux
 - Python 3.6
 - FFmpeg

Similar environments (e.g. with OSX, Python 2) might work with small modification, but not tested.

## Data loader

This data loader that directly takes a compressed video and returns compressed representation (I-frame, motion vectors, or residual) as a numpy array.

#### Supported video format
Currently we only support mpeg4 raw videos. Other codecs, e.g. H.264, coming soon. The mpeg4 raw videos can be obtained using FFmpeg:

`ffmpeg -i input.mp4 -c:v  -c:v mpeg4 -f rawvideo output.mp4`

#### Install
 - Download FFmpeg (`git clone https://github.com/FFmpeg/FFmpeg.git`).
 - Go to FFmpeg home,  and `git checkout 74c6a6d3735f79671b177a0e0c6f2db696c2a6d2`.
 - `make clean`
 - `./configure --prefix=${FFMPEG_INSTALL_PATH} --enable-pic --disable-yasm --enable-shared`
 - `make`
 - `make install`
 - If needed, add `${FFMPEG_INSTALL_PATH}/lib/` to `$LD_LIBRARY_PATH`.
 - Go to `data_loader` folder.
 - Modify `setup.py` to use your FFmpeg path (`${FFMPEG_INSTALL_PATH}`).
 - `./install.sh`

#### Usage
The data loader has two functions: `load` for loading a representation and `get_num_frames` for
counting the number of frames in a video.

The following call returns one frame (specified by `frame_index=0,1,...`) of one GOP
(specified by `gop_index=0,1,...`).
```python
from coviar import load
load([input], [gop_index], [frame_index], [representation_type], [accumulate])
```
 - input: path to video (.mp4).
 - representation_type: `0`, `1`, or `2`. `0` for I-frames, `1` for motion vectors, `2` for residuals.
 - accumulate: `True` or `False`. `True` returns the accumulated representation. `False` returns the original compressed representations. (See paper for details. )

For example, 
```
load(input.mp4, 3, 8, 1, True)
```
returns the accumulated motion vectors of the 9th frame of the 4th GOP.
