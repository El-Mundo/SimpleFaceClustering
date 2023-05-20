# SimpleFaceClustering
A very basic Face Clustering program based on DeepFace

## Usage

This program gives credit to the powerful library of [DeepFace](https://github.com/serengil/deepface).
To use this program, please see the [Demo](demo.ipynb).

It also comes with a JAVA-Processing-4-based [visualisation program](viz/viz.pde), which looks like this (in ideal conditions):

![VisualisedFaceClusters](img_demo/sample.png)

## Controls

Keys for controlling the visualisation program:

### Export
Press key 'e' to export the results of clusters

### Display
The axis scaling factor can be controlled with keys 'a' and 'd'.
The local scaling of images can be controlled with keys 's' and 'w'.
The zooming of the camera can be controlled with the mouse wheel.
The camera can be moved by mouse dragging..

### Similarity Comparing
The threshold that determines the how close can two persons be considered of the same identity can be controlled with keys 'j' and 'k'. Lines will be drawn between the faces that are believed to be of the same identity.