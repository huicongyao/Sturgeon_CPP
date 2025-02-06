//
// Created by yaohc on 2024/12/20.
//
#include "utils.h"
#include "DataLoader/DataLoader.h"

hsize_t Sturgeon::iterateHDF5(const H5::Group &group, const fs::path &path) {
    hsize_t numObjs = group.getNumObjs();
    long long total_chunks = 0;
    for (hsize_t i = 0; i < numObjs; ++i) {
        // Get the name of the object
        std::string objName = group.getObjnameByIdx(i);

        // Get the type of the object (group or dataset)
        H5G_obj_t objType = group.getObjTypeByIdx(i);

        if (objType == H5G_GROUP) {
            // If it's a group, open it and iterate recursively
            H5::Group subGroup = group.openGroup(objName);
            iterateHDF5(subGroup);

        } else if (objType == H5G_DATASET) {
            // If it's a dataset, open it and process
            H5::DataSet dataset = group.openDataSet(objName);
            // Example: read raw data if it's of integer type
            H5::DataSpace dataspace = dataset.getSpace();
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, nullptr);
            Sturgeon::Signal signal(path.filename(), objName, {static_cast<long>(dims[0])});
            dataset.read(signal.data.data_ptr(), H5::PredType::NATIVE_FLOAT);
            auto chunks = signal.get_chunks(6000, 5, 500);
            total_chunks += chunks.size();

//            for (auto &Chunk: chunks) {
//                std::cout << Chunk.data << std::endl;
//            }
        }
    }
    return total_chunks;
}

std::vector<fs::path> Sturgeon::get_h5_files(const fs::path &path) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".h5") {
            files.push_back(entry.path());
        }
    }
    return files;
}

int32_t
Sturgeon::compute_trim_start(const torch::Tensor &signal,
                             int32_t window_size,
                             float threshold,
                             int32_t min_trim,
                             int32_t min_elements,
                             int32_t max_samples,
                             float max_trim) {
    bool seen_peak = false;
    int32_t signal_length = signal.size(0);
    int32_t num_windows = std::min(max_samples, signal_length) / window_size;

    for (int pos = 0; pos < num_windows; ++pos) {
        int32_t start = pos * window_size + min_trim;
        int32_t end = start + window_size;

        if (end > signal_length) break;

        torch::Tensor window = signal.slice(0, start, end);
        int32_t count_above_threshold = window.gt(threshold).sum().item<int32_t>();

        if (count_above_threshold >= min_elements || seen_peak) {
            seen_peak = true;

            // Check if the last element of the window is greater than the threshold
            if (window[-1].item<float>() > threshold) continue;

            // Check if we are exceeding the max samples or max trim limits
            if (end >= std::min(max_samples, signal_length) || static_cast<float>(end) / signal_length > max_trim) {
                return min_trim; // Return the default minimum trim value
            }
            return end;
        }
    }
    return min_trim;
}

std::pair<float, float> Sturgeon::get_norm_factors(const torch::Tensor &signal, std::string method) {
    float s_scale, s_shift;
    if (method == "mad") {
        float c = 0.674489;
        s_shift = torch::median(signal).item<float>();
        s_scale = torch::median(torch::abs(signal - s_shift)).item<float>() / c;
    } else if (method == "mean") {
        s_scale = torch::mean(signal).item<float>();
        s_shift = torch::std(signal).item<float>();
    } else {
        throw std::invalid_argument("Invalid normalization method, either use median absolute deviation (mad) or mean absolute deviation (mean)");
    }
    return std::make_pair(s_shift, s_scale);
}
