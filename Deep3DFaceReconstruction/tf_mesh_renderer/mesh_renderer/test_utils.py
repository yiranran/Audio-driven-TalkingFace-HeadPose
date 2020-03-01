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

"""Common functions for the rasterizer and mesh renderer tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf


def check_jacobians_are_nearly_equal(theoretical,
                                     numerical,
                                     outlier_relative_error_threshold,
                                     max_outlier_fraction,
                                     include_jacobians_in_error_message=False):
  """Compares two Jacobian matrices, allowing for some fraction of outliers.

  Args:
    theoretical: 2D numpy array containing a Jacobian matrix with entries
        computed via gradient functions. The layout should be as in the output
        of gradient_checker.
    numerical: 2D numpy array of the same shape as theoretical containing a
        Jacobian matrix with entries computed via finite difference
        approximations. The layout should be as in the output
        of gradient_checker.
    outlier_relative_error_threshold: float prescribing the maximum relative
        error (from the finite difference approximation) is tolerated before
        and entry is considered an outlier.
    max_outlier_fraction: float defining the maximum fraction of entries in
        theoretical that may be outliers before the check returns False.
    include_jacobians_in_error_message: bool defining whether the jacobian
        matrices should be included in the return message should the test fail.

  Returns:
    A tuple where the first entry is a boolean describing whether
    max_outlier_fraction was exceeded, and where the second entry is a string
    containing an error message if one is relevant.
  """
  outlier_gradients = np.abs(
      numerical - theoretical) / numerical > outlier_relative_error_threshold
  outlier_fraction = np.count_nonzero(outlier_gradients) / np.prod(
      numerical.shape[:2])
  jacobians_match = outlier_fraction <= max_outlier_fraction

  message = (
      ' %f of theoretical gradients are relative outliers, but the maximum'
      ' allowable fraction is %f ' % (outlier_fraction, max_outlier_fraction))
  if include_jacobians_in_error_message:
    # the gradient_checker convention is the typical Jacobian transposed:
    message += ('\nNumerical Jacobian:\n%s\nTheoretical Jacobian:\n%s' %
                (repr(numerical.T), repr(theoretical.T)))
  return jacobians_match, message


def expect_image_file_and_render_are_near(test_instance,
                                          sess,
                                          baseline_path,
                                          result_image,
                                          max_outlier_fraction=0.001,
                                          pixel_error_threshold=0.01):
  """Compares the output of mesh_renderer with an image on disk.

  The comparison is soft: the images are considered identical if at most
  max_outlier_fraction of the pixels differ by more than a relative error of
  pixel_error_threshold of the full color value. Note that before comparison,
  mesh renderer values are clipped to the range [0,1].

  Uses _images_are_near for the actual comparison.

  Args:
    test_instance: a python unit test instance.
    sess: a TensorFlow session for decoding the png.
    baseline_path: path to the reference image on disk.
    result_image: the result image, as a numpy array.
    max_outlier_fraction: the maximum fraction of outlier pixels allowed.
    pixel_error_threshold: pixel values are considered to differ if their
      difference exceeds this amount. Range is 0.0 - 1.0.
  """
  baseline_bytes = open(baseline_path, 'rb').read()
  baseline_image = sess.run(tf.image.decode_png(baseline_bytes))

  test_instance.assertEqual(baseline_image.shape, result_image.shape,
                            'Image shapes %s and %s do not match.' %
                            (baseline_image.shape, result_image.shape))

  result_image = np.clip(result_image, 0., 1.).copy(order='C')
  baseline_image = baseline_image.astype(float) / 255.0

  outlier_channels = (np.abs(baseline_image - result_image) >
                      pixel_error_threshold)
  outlier_pixels = np.any(outlier_channels, axis=2)
  outlier_count = np.count_nonzero(outlier_pixels)
  outlier_fraction = outlier_count / np.prod(baseline_image.shape[:2])
  images_match = outlier_fraction <= max_outlier_fraction

  outputs_dir = "/tmp" #os.environ["TEST_TMPDIR"]
  base_prefix = os.path.splitext(os.path.basename(baseline_path))[0]
  result_output_path = os.path.join(outputs_dir, base_prefix + "_result.png")

  message = ('%s does not match. (%f of pixels are outliers, %f is allowed.). '
             'Result image written to %s' %
             (baseline_path, outlier_fraction, max_outlier_fraction, result_output_path))

  if not images_match:
    result_bytes = sess.run(tf.image.encode_png(result_image*255.0))
    with open(result_output_path, 'wb') as output_file:
      output_file.write(result_bytes)

  test_instance.assertTrue(images_match, msg=message)
