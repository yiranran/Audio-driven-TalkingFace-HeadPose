package(default_visibility = ["//visibility:public"])

py_library(
    name = "mesh_renderer",
    srcs = ["mesh_renderer.py"],
    deps = [
        ":rasterize_triangles",
    ],
)

py_test(
    name = "mesh_renderer_test",
    size = "medium",
    srcs = ["mesh_renderer_test.py"],
    data = [
        "//mesh_renderer/test_data:images",
        "//mesh_renderer/kernels:rasterize_triangles_kernel",
    ],
    deps = [
        ":mesh_renderer",
    ],
)

py_library(
    name = "camera_utils",
    srcs = ["camera_utils.py"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
)

py_library(
    name = "rasterize_triangles",
    srcs = ["rasterize_triangles.py"],
    deps = [
        ":camera_utils",
    ],
)

py_test(
    name = "rasterize_triangles_test",
    srcs = ["rasterize_triangles_test.py"],
    data = [
        "//mesh_renderer/test_data:images",
        "//mesh_renderer/kernels:rasterize_triangles_kernel",
    ],
    deps = [
        ":camera_utils",
        ":rasterize_triangles",
        ":test_utils",
    ],
)
