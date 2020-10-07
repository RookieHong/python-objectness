#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "image.h"
#include <vector>
#include "segment-image.h"

namespace py = pybind11;

// -------------
// pure C++ code
// -------------

std::vector<int> segment_img(const std::vector<uint8_t> &img_vec, const int height, const int width, const double sigma, const double k, const int min_size)
{
    // Assume the img_vec is from an rgb image.
    image<int> *seg;
    image<rgb> *im = new image<rgb>(width, height);
    rgb *imPtr = im->data;
    int num_elem = height * width;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            rgb pixel;
            int start_idx = (y * width + x) * 3;
            pixel.r = img_vec[start_idx];
            pixel.g = img_vec[start_idx + 1];
            pixel.b = img_vec[start_idx + 2];
            *(imPtr++) = pixel;
        }
    }

    int num_ccs;
    seg = segment_image_int(im, sigma, k, min_size, &num_ccs);
    delete im;

    int *seg_ptr = seg->data;
    std::vector<int> output(num_elem);  // output is an H x W image.
    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            output[y * width + x] = *(seg_ptr++);
        }
    }

    delete seg;

    return output;
}

// ----------------
// Python interface
// ----------------

// wrap C++ function with NumPy array IO
py::array_t<uint8_t> py_(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> flatten_img, const int height, const int width, const double sigma, const double k, const int min_size)
{
    // allocate std::vector (to pass to the C++ function)
    std::vector<uint8_t> img_vec(flatten_img.size());

    // copy py::array -> std::vector
    std::memcpy(img_vec.data(), flatten_img.data(), flatten_img.size() * sizeof(uint8_t));

    // call pure C++ function
    std::vector<int> result_vec = segment_img(img_vec, height, width, sigma, k, min_size);

    // allocate py::array (to pass the result of the C++ function to Python)
    auto result = py::array_t<int>(result_vec.size());
    auto result_buffer = result.request();
    int *result_ptr = (int *)result_buffer.ptr;

    // copy std::vector -> py::array
    std::memcpy(result_ptr, result_vec.data(), result_vec.size() * sizeof(int));

    return result;
}

// wrap as Python module
PYBIND11_MODULE(c_segment, m)
{
    m.doc() = "Segment an rgb image into superpixels.";

    m.def("segment_img", &segment_img, "Segment an rgb image into superpixels.");
}