#ifndef PYTHON_VMAPPING_H
#define PYTHON_VMAPPING_H

#include <memory>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "vmapping.h"

namespace py = pybind11;

class VoxelitPython
{
public:
    VoxelitPython(int w, int h, const Eigen::Matrix3f &K);
    ~VoxelitPython();

    void reset();
    void createMap(int numEntries, int numVoxels, float voxelSize);
    void fuseDepth(cv::Mat depth, const Eigen::Matrix4f &pose);
    void loadAndFuseDepth(std::string depth, const Eigen::Matrix4f &pose);
    py::tuple getPlygonMesh();
    std::vector<Eigen::Vector3f> getSurfacePoints();
    void setDepthScale(float scale);

private:
    float depth_scale;
    std::shared_ptr<vmap::VoxelMapping> voxel_map;
};

#endif // PYTHON_VMAPPING_H