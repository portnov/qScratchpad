qScratchpad README
==================

qScratchpad intends to be an application for very fast doodling, especially,
for scratching all sorts of "diagrams" and "charts" (well, actually doodles)
you currently write with pen on paper while thinking about things like
algorithms or software architecture or customer requirements.

For most people, it is not convinient to draw with mouse, so primarily this
application is designed to work with any sort of tablet. Although mouse is also
supported.

Key features:

* Fast enough
* Potentially infinite canvas
* Simple panning with middle mouse/tablet button, and zooming with mouse/tablet wheel
* Own file format (`json.gz`) for loading/saving; support for exporting PNG files.

The idea is to let you think about whatever do you want to think about, and not
about "should I draw this line red instead of blue" or "these boxes are not
alighed nice enough". For this, only simple drawing features are available: you
just draw lines, currently even without ability to fill them.

Main object that qScratchpad allows you to draw is Bezier splines - but do not
worry, it does not ask you to manually move the control points. Instead, you
just draw, and the application draws the curve. Bezier logic is used mainly for
smoothing.

Apart from smooth curves, you can also draw "straightened" curves, line
segments, circles and rectangles. The drawing method is also specialized for
fast doodling, not for precision: you manually draw, for example, something
roughly resembling a rectangle, and qScratchpad immediately replaces it with a
nice rectangle. There are explicit modes for all supported kinds of figures,
when you press, for example, "Circle", and draw circles. Also there is an
"Auto" mode, when you let qScratchpad decide, what your doodles remind it
about.

![demo](https://user-images.githubusercontent.com/284644/155070924-8a6f4b38-d175-4cad-9791-4d163763208c.png)

Prerequisites
-------------

* Python 3.7+
* PyQt5

Running
-------

    $ ./qscratchpad.py

License
-------

License: GPLv3.

