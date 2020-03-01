This is a differentiable, 3D mesh renderer using TensorFlow.

This is not an official Google product.

The interface to the renderer is provided by mesh_renderer.py and
rasterize_triangles.py, which provide TensorFlow Ops that can be added to a
TensorFlow graph. The internals of the renderer are handled by a C++ kernel.

The input to the C++ rendering kernel is a list of 3D vertices and a list of
triangles, where a triangle consists of a list of three vertex ids. The
output of the renderer is a pair of images containing triangle ids and
barycentric weights. Pixel values in the barycentric weight image are the
weights of the pixel center point with respect to the triangle at that pixel
(identified by the triangle id). The renderer provides derivatives of the
barycentric weights of the pixel centers with respect to the vertex
positions.

Any approximation error stems from the assumption that the triangle id at a
pixel does not change as the vertices are moved. This is a reasonable
approximation for small changes in vertex position. Even when the triangle id
does change, the derivatives will be computed by extrapolating the barycentric
weights of a neighboring triangle, which will produce a good approximation if
the mesh is smooth. The main source of error occurs at occlusion boundaries, and
particularly at the edge of an open mesh, where the background appears opposite
the triangle's edge.

The algorithm implemented is described by Olano and Greer, "Triangle Scan
Conversion using 2D Homogeneous Coordinates," HWWS 1997.

How to Build
------------

Follow the instructions to [install TensorFlow using virtualenv](https://www.tensorflow.org/install/install_linux#installing_with_virtualenv).

Build and run tests using Bazel from inside the (tensorflow) virtualenv:

`(tensorflow)$ ./runtests.sh`

The script calls the Bazel rules using the Python interpreter at
`$VIRTUAL_ENV/bin/python`. If you aren't using virtualenv, `bazel test ...` may
be sufficient.

Citation
--------

If you use this renderer in your research, please cite [this paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Genova_Unsupervised_Training_for_CVPR_2018_paper.html "CVF Version"):

*Unsupervised Training for 3D Morphable Model Regression*. Kyle Genova, Forrester Cole, Aaron Maschinot, Aaron Sarna, Daniel Vlasic, and William T. Freeman. CVPR 2018, pp. 8377-8386.

```
@InProceedings{Genova_2018_CVPR,
  author = {Genova, Kyle and Cole, Forrester and Maschinot, Aaron and Sarna, Aaron and Vlasic, Daniel and Freeman, William T.},
  title = {Unsupervised Training for 3D Morphable Model Regression},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}
```
