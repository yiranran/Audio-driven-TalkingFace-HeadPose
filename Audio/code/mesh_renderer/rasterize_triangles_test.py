# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import test_utils
import camera_utils
import rasterize_triangles


class RenderTest(tf.test.TestCase):

  def setUp(self):
    self.test_data_directory = 'mesh_renderer/test_data/'

    tf.reset_default_graph()
    self.cube_vertex_positions = tf.constant(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
         [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=tf.float32)
    self.cube_triangles = tf.constant(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=tf.int32)

    self.tf_float = lambda x: tf.constant(x, dtype=tf.float32)

    self.image_width = 640
    self.image_height = 480

    self.perspective = camera_utils.perspective(
        self.image_width / self.image_height,
        self.tf_float([40.0]), self.tf_float([0.01]),
        self.tf_float([10.0]))

  def runTriangleTest(self, w_vector, target_image_name):
    """Directly renders a rasterized triangle's barycentric coordinates.

    Tests only the kernel (rasterize_triangles_module).

    Args:
      w_vector: 3 element vector of w components to scale triangle vertices.
      target_image_name: image file name to compare result against.
    """
    clip_init = np.array(
        [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
        dtype=np.float32)
    clip_init = clip_init * np.reshape(
        np.array(w_vector, dtype=np.float32), [3, 1])

    clip_coordinates = tf.constant(clip_init)
    triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

    rendered_coordinates, _, _ = (
        rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
            clip_coordinates, triangles, self.image_width, self.image_height))
    rendered_coordinates = tf.concat(
        [rendered_coordinates,
         tf.ones([self.image_height, self.image_width, 1])], axis=2)
    with self.test_session() as sess:
      image = rendered_coordinates.eval()
      baseline_image_path = os.path.join(self.test_data_directory,
                                         target_image_name)
      test_utils.expect_image_file_and_render_are_near(
          self, sess, baseline_image_path, image)

  def testRendersSimpleTriangle(self):
    self.runTriangleTest((1.0, 1.0, 1.0), 'Simple_Triangle.png')

  def testRendersPerspectiveCorrectTriangle(self):
    self.runTriangleTest((0.2, 0.5, 2.0), 'Perspective_Corrected_Triangle.png')

  def testRendersTwoCubesInBatch(self):
      """Renders a simple cube in two viewpoints to test the python wrapper."""

      vertex_rgb = (self.cube_vertex_positions * 0.5 + 0.5)
      vertex_rgba = tf.concat([vertex_rgb, tf.ones([8, 1])], axis=1)

      center = self.tf_float([[0.0, 0.0, 0.0]])
      world_up = self.tf_float([[0.0, 1.0, 0.0]])
      look_at_1 = camera_utils.look_at(self.tf_float([[2.0, 3.0, 6.0]]),
          center, world_up)
      look_at_2 = camera_utils.look_at(self.tf_float([[-3.0, 1.0, 6.0]]),
          center, world_up)
      projection_1 = tf.matmul(self.perspective, look_at_1)
      projection_2 = tf.matmul(self.perspective, look_at_2)
      projection = tf.concat([projection_1, projection_2], axis=0)
      background_value = [0.0, 0.0, 0.0, 0.0]

      rendered = rasterize_triangles.rasterize(
          tf.stack([self.cube_vertex_positions, self.cube_vertex_positions]),
          tf.stack([vertex_rgba, vertex_rgba]), self.cube_triangles, projection,
          self.image_width, self.image_height, background_value)

      with self.test_session() as sess:
        images = sess.run(rendered, feed_dict={})
        for i in (0, 1):
          image = images[i, :, :, :]
          baseline_image_name = 'Unlit_Cube_{}.png'.format(i)
          baseline_image_path = os.path.join(self.test_data_directory,
                                            baseline_image_name)
          test_utils.expect_image_file_and_render_are_near(
            self, sess, baseline_image_path, image)

  def testSimpleTriangleGradientComputation(self):
    """Verifies the Jacobian matrix for a single pixel.

    The pixel is in the center of a triangle facing the camera. This makes it
    easy to check which entries of the Jacobian might not make sense without
    worrying about corner cases.
    """
    test_pixel_x = 325
    test_pixel_y = 245

    clip_coordinates = tf.placeholder(tf.float32, shape=[3, 4])

    triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

    barycentric_coordinates, _, _ = (
        rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
            clip_coordinates, triangles, self.image_width, self.image_height))

    pixels_to_compare = barycentric_coordinates[
        test_pixel_y:test_pixel_y + 1, test_pixel_x:test_pixel_x + 1, :]

    with self.test_session():
      ndc_init = np.array(
          [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
          dtype=np.float32)
      theoretical, numerical = tf.test.compute_gradient(
          clip_coordinates, (3, 4),
          pixels_to_compare, (1, 1, 3),
          x_init_value=ndc_init,
          delta=4e-2)
      jacobians_match, message = (
          test_utils.check_jacobians_are_nearly_equal(
              theoretical, numerical, 0.01, 0.0, True))
      self.assertTrue(jacobians_match, message)

  def testInternalRenderGradientComputation(self):
    """Isolates and verifies the Jacobian matrix for the custom kernel."""
    image_height = 21
    image_width = 28

    clip_coordinates = tf.placeholder(tf.float32, shape=[8, 4])

    barycentric_coordinates, _, _ = (
        rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
            clip_coordinates, self.cube_triangles, image_width, image_height))

    with self.test_session():
      # Precomputed transformation of the simple cube to normalized device
      # coordinates, in order to isolate the rasterization gradient.
      # pyformat: disable
      ndc_init = np.array(
          [[-0.43889722, -0.53184521, 0.85293502, 1.0],
           [-0.37635487, 0.22206162, 0.90555805, 1.0],
           [-0.22849123, 0.76811147, 0.80993629, 1.0],
           [-0.2805393, -0.14092168, 0.71602166, 1.0],
           [0.18631913, -0.62634289, 0.88603103, 1.0],
           [0.16183566, 0.08129397, 0.93020856, 1.0],
           [0.44147962, 0.53497446, 0.85076219, 1.0],
           [0.53008741, -0.31276882, 0.77620775, 1.0]],
          dtype=np.float32)
      # pyformat: enable
      theoretical, numerical = tf.test.compute_gradient(
          clip_coordinates, (8, 4),
          barycentric_coordinates, (image_height, image_width, 3),
          x_init_value=ndc_init,
          delta=4e-2)
      jacobians_match, message = (
          test_utils.check_jacobians_are_nearly_equal(
              theoretical, numerical, 0.01, 0.01))
      self.assertTrue(jacobians_match, message)


if __name__ == '__main__':
  tf.test.main()
