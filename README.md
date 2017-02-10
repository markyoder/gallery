# gallery
A gallery of sample codes, show-pieces, and demos.

Cloning:

git clone --recursive https://github.com/markyoder/gallery.git
If you forget the --recursive, you won't get the submodules. fix it like:

git submodule update --init --recursive which tells git to run two commands:

git submodule init ... and subsequently, git submodule update
We're working on making this happen as smoothly as possible, but it may still be necessary to install a few extra bits. We recommend using Anaconda Python 3.x; for those stubborn amongst us who insist to run on their system Python, you an probably just replace "conda" installations with "pip":

on a fresh linux install... stuff we have to do besides just clone this:

- pip install geopy
- conda install basemap
- pip install geographiclib
- conda install -c ioos rtree
