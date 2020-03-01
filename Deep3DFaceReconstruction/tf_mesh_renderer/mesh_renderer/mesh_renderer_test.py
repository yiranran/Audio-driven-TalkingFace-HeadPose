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

import math
import os

import numpy as np
import tensorflow as tf

import camera_utils
import mesh_renderer
import test_utils


class RenderTest(tf.test.TestCase):

  def setUp(self):
    self.test_data_directory = (
        'mesh_renderer/test_data/')

    tf.reset_default_graph()
    # Set up a basic cube centered at the origin, with vertex normals pointing
    # outwards along the line from the origin to the cube vertices:
    self.cube_vertices = tf.constant(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
         [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=tf.float32)
    self.cube_normals = tf.nn.l2_normalize(self.cube_vertices, dim=1)
    self.cube_triangles = tf.constant(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=tf.int32)

  def testRendersSimpleCube(self):
    """Renders a simple cube to test the full forward pass.

    Verifies the functionality of both the custom kernel and the python wrapper.
    """

    model_transforms = camera_utils.euler_matrices(
        [[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]])[:, :3, :3]

    vertices_world_space = tf.matmul(
        tf.stack([self.cube_vertices, self.cube_vertices]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([self.cube_normals, self.cube_normals]),
        model_transforms,
        transpose_b=True)

    # camera position:
    eye = tf.constant(2 * [[0.0, 0.0, 6.0]], dtype=tf.float32)
    center = tf.constant(2 * [[0.0, 0.0, 0.0]], dtype=tf.float32)
    world_up = tf.constant(2 * [[0.0, 1.0, 0.0]], dtype=tf.float32)
    image_width = 640
    image_height = 480
    light_positions = tf.constant([[[0.0, 0.0, 6.0]], [[0.0, 0.0, 6.0]]])
    light_intensities = tf.ones([2, 1, 3], dtype=tf.float32)
    vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)

    rendered = mesh_renderer.mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)

    with self.test_session() as sess:
      images = sess.run(rendered, feed_dict={})
      for image_id in range(images.shape[0]):
        target_image_name = 'Gray_Cube_%i.png' % image_id
        baseline_image_path = os.path.join(self.test_data_directory,
                                           target_image_name)
        test_utils.expect_image_file_and_render_are_near(
            self, sess, baseline_image_path, images[image_id, :, :, :])

  def testComplexShading(self):
    """Tests specular highlights, colors, and multiple lights per image."""
    # rotate the cube for the test:
    model_transforms = camera_utils.euler_matrices(
        [[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]])[:, :3, :3]

    vertices_world_space = tf.matmul(
        tf.stack([self.cube_vertices, self.cube_vertices]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([self.cube_normals, self.cube_normals]),
        model_transforms,
        transpose_b=True)

    # camera position:
    eye = tf.constant([[0.0, 0.0, 6.0], [0., 0.2, 18.0]], dtype=tf.float32)
    center = tf.constant([[0.0, 0.0, 0.0], [0.1, -0.1, 0.1]], dtype=tf.float32)
    world_up = tf.constant(
        [[0.0, 1.0, 0.0], [0.1, 1.0, 0.15]], dtype=tf.float32)
    fov_y = tf.constant([40., 13.3], dtype=tf.float32)
    near_clip = tf.constant(0.1, dtype=tf.float32)
    far_clip = tf.constant(25.0, dtype=tf.float32)
    image_width = 640
    image_height = 480
    light_positions = tf.constant([[[0.0, 0.0, 6.0], [1.0, 2.0, 6.0]],
                                   [[0.0, -2.0, 4.0], [1.0, 3.0, 4.0]]])
    light_intensities = tf.constant(
        [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[2.0, 0.0, 1.0], [0.0, 2.0,
                                                                1.0]]],
        dtype=tf.float32)
    # pyformat: disable
    vertex_diffuse_colors = tf.constant(2*[[[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                            [1.0, 1.0, 1.0],
                                            [1.0, 1.0, 0.0],
                                            [1.0, 0.0, 1.0],
                                            [0.0, 1.0, 1.0],
                                            [0.5, 0.5, 0.5]]],
                                        dtype=tf.float32)
    vertex_specular_colors = tf.constant(2*[[[0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [1.0, 1.0, 1.0],
                                             [1.0, 1.0, 0.0],
                                             [1.0, 0.0, 1.0],
                                             [0.0, 1.0, 1.0],
                                             [0.5, 0.5, 0.5],
                                             [1.0, 0.0, 0.0]]],
                                         dtype=tf.float32)
    # pyformat: enable
    shininess_coefficients = 6.0 * tf.ones([2, 8], dtype=tf.float32)
    ambient_color = tf.constant(
        [[0., 0., 0.], [0.1, 0.1, 0.2]], dtype=tf.float32)
    renders = mesh_renderer.mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height, vertex_specular_colors,
        shininess_coefficients, ambient_color, fov_y, near_clip, far_clip)
    tonemapped_renders = tf.concat(
        [
            mesh_renderer.tone_mapper(renders[:, :, :, 0:3], 0.7),
            renders[:, :, :, 3:4]
        ],
        axis=3)

    # Check that shininess coefficient broadcasting works by also rendering
    # with a scalar shininess coefficient, and ensuring the result is identical:
    broadcasted_renders = mesh_renderer.mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height, vertex_specular_colors,
        6.0, ambient_color, fov_y, near_clip, far_clip)
    tonemapped_broadcasted_renders = tf.concat(
        [
            mesh_renderer.tone_mapper(broadcasted_renders[:, :, :, 0:3], 0.7),
            broadcasted_renders[:, :, :, 3:4]
        ],
        axis=3)

    with self.test_session() as sess:
      images, broadcasted_images = sess.run(
          [tonemapped_renders, tonemapped_broadcasted_renders], feed_dict={})

      for image_id in range(images.shape[0]):
        target_image_name = 'Colored_Cube_%i.png' % image_id
        baseline_image_path = os.path.join(self.test_data_directory,
                                           target_image_name)
        test_utils.expect_image_file_and_render_are_near(
            self, sess, baseline_image_path, images[image_id, :, :, :])
        test_utils.expect_image_file_and_render_are_near(
            self, sess, baseline_image_path,
            broadcasted_images[image_id, :, :, :])

  def testFullRenderGradientComputation(self):
    """Verifies the Jacobian matrix for the entire renderer.

    This ensures correct gradients are propagated backwards through the entire
    process, not just through the rasterization kernel. Uses the simple cube
    forward pass.
    """
    image_height = 21
    image_width = 28

    # rotate the cube for the test:
    model_transforms = camera_utils.euler_matrices(
        [[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]])[:, :3, :3]

    vertices_world_space = tf.matmul(
        tf.stack([self.cube_vertices, self.cube_vertices]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([self.cube_normals, self.cube_normals]),
        model_transforms,
        transpose_b=True)

    # camera position:
    eye = tf.constant([0.0, 0.0, 6.0], dtype=tf.float32)
    center = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    world_up = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)

    # Scene has a single light from the viewer's eye.
    light_positions = tf.expand_dims(tf.stack([eye, eye], axis=0), axis=1)
    light_intensities = tf.ones([2, 1, 3], dtype=tf.float32)

    vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)

    rendered = mesh_renderer.mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)

    with self.test_session():
      theoretical, numerical = tf.test.compute_gradient(
          self.cube_vertices, (8, 3),
          rendered, (2, image_height, image_width, 4),
          x_init_value=self.cube_vertices.eval(),
          delta=1e-3)
      jacobians_match, message = (
          test_utils.check_jacobians_are_nearly_equal(
              theoretical, numerical, 0.01, 0.01))
      self.assertTrue(jacobians_match, message)

  def testThatCubeRotates(self):
    """Optimize a simple cube's rotation using pixel loss.

    The rotation is represented as static-basis euler angles. This test checks
    that the computed gradients are useful.
    """
    image_height = 480
    image_width = 640
    initial_euler_angles = [[0.0, 0.0, 0.0]]

    euler_angles = tf.Variable(initial_euler_angles)
    model_rotation = camera_utils.euler_matrices(euler_angles)[0, :3, :3]

    vertices_world_space = tf.reshape(
        tf.matmul(self.cube_vertices, model_rotation, transpose_b=True),
        [1, 8, 3])

    normals_world_space = tf.reshape(
        tf.matmul(self.cube_normals, model_rotation, transpose_b=True),
        [1, 8, 3])

    # camera position:
    eye = tf.constant([[0.0, 0.0, 6.0]], dtype=tf.float32)
    center = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    world_up = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)

    vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)
    light_positions = tf.reshape(eye, [1, 1, 3])
    light_intensities = tf.ones([1, 1, 3], dtype=tf.float32)

    render = mesh_renderer.mesh_renderer(
        vertices_world_space, self.cube_triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)
    render = tf.reshape(render, [image_height, image_width, 4])

    # Pick the desired cube rotation for the test:
    test_model_rotation = camera_utils.euler_matrices([[-20.0, 0.0,
                                                        60.0]])[0, :3, :3]

    desired_vertex_positions = tf.reshape(
        tf.matmul(self.cube_vertices, test_model_rotation, transpose_b=True),
        [1, 8, 3])
    desired_normals = tf.reshape(
        tf.matmul(self.cube_normals, test_model_rotation, transpose_b=True),
        [1, 8, 3])
    desired_render = mesh_renderer.mesh_renderer(
        desired_vertex_positions, self.cube_triangles, desired_normals,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)
    desired_render = tf.reshape(desired_render, [image_height, image_width, 4])

    loss = tf.reduce_mean(tf.abs(render - desired_render))
    optimizer = tf.train.MomentumOptimizer(0.7, 0.1)
    grad = tf.gradients(loss, [euler_angles])
    grad, _ = tf.clip_by_global_norm(grad, 1.0)
    opt_func = optimizer.apply_gradients([(grad[0], euler_angles)])

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(35):
        sess.run([loss, opt_func])
      final_image, desired_image = sess.run([render, desired_render])

      target_image_name = 'Gray_Cube_0.png'
      baseline_image_path = os.path.join(self.test_data_directory,
                                         target_image_name)
      test_utils.expect_image_file_and_render_are_near(
          self, sess, baseline_image_path, desired_image)
      test_utils.expect_image_file_and_render_are_near(
          self,
          sess,
          baseline_image_path,
          final_image,
          max_outlier_fraction=0.01,
          pixel_error_threshold=0.04)


if __name__ == '__main__':
  tf.test.main()
