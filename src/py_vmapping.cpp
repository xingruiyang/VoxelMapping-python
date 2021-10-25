#include <opencv2/opencv.hpp>

#include "py_vmapping.h"
#include "ndarray_converter.h"

namespace py = pybind11;

VoxelitPython::VoxelitPython(int w, int h, const Eigen::Matrix3f &K)
    : depth_scale(1000),
      voxel_map(new vmap::VoxelMapping(w, h, K))
{
}

VoxelitPython::~VoxelitPython()
{
}

void VoxelitPython::reset()
{
    voxel_map->reset();
}

void VoxelitPython::createMap(int numEntries, int numVoxels, float voxelSize)
{
    voxel_map->CreateMap(numEntries, numVoxels, voxelSize);
}

void VoxelitPython::fuseDepth(cv::Mat depth, const Eigen::Matrix4f &pose)
{
    cv::Mat depthF;
    depth.convertTo(depthF, CV_32FC1, 1 / depth_scale);
    voxel_map->FuseDepth(cv::cuda::GpuMat(depthF), pose);
}

void VoxelitPython::loadAndFuseDepth(std::string depth, const Eigen::Matrix4f &pose)
{
    cv::Mat depthF;
    cv::Mat img = cv::imread(depth, -1);
    img.convertTo(depthF, CV_32FC1, 1 / depth_scale);
    voxel_map->FuseDepth(cv::cuda::GpuMat(depthF), pose);
}

cv::Mat VoxelitPython::getDepthMap(const Eigen::Matrix4f &pose)
{
    cv::cuda::GpuMat vmap;
    voxel_map->RenderScene(vmap, pose);
    return cv::Mat(vmap);
}

py::tuple VoxelitPython::getPlygonMesh()
{
    float *verts, *norms;
    size_t num_tri = voxel_map->Polygonize(verts, norms);
    return py::make_tuple(
        py::array_t<float>(num_tri * 9, verts),
        py::array_t<float>(num_tri * 9, norms));
}

std::vector<Eigen::Vector3f> VoxelitPython::getSurfacePoints()
{
    return voxel_map->GetSurfacePoints();
}

void VoxelitPython::setDepthScale(float scale)
{
    depth_scale = scale;
}

PYBIND11_MODULE(py_vmapping, m)
{
    NDArrayConverter::init_numpy();
    py::class_<VoxelitPython>(m, "map")
        .def(py::init<int, int, const Eigen::Matrix3f &>())
        .def("reset", &VoxelitPython::reset)
        .def("create_map", &VoxelitPython::createMap)
        .def("fuse_depth", &VoxelitPython::fuseDepth)
        .def("get_depth", &VoxelitPython::getDepthMap)
        .def("get_polygon", &VoxelitPython::getPlygonMesh)
        .def("get_surface_points", &VoxelitPython::getSurfacePoints)
        .def("set_depth_scale", &VoxelitPython::setDepthScale)
        .def("load_and_fuse_depth", &VoxelitPython::loadAndFuseDepth);
}