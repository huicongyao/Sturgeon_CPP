#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/map.hpp>
#include <sstream>
#include <unordered_map>
#include <map>
#include <vector>
#include <chrono>
#include <iostream>
#include <H5Cpp.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <chrono>
#include <thread>

namespace bip = boost::interprocess;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: child_program <shared_memory_name> <file_path>" << std::endl;
        return 1;
    }
    const std::string shared_memory_name = argv[1];
    const std::string file_path = argv[2];

    auto st1 = std::chrono::high_resolution_clock::now();
    H5::H5Library::open(); // 确保HDF5已初始化
    H5::FileAccPropList fapl;
    fapl.setFcloseDegree(H5F_CLOSE_STRONG);
    fapl.setCache(0, 1024, 300 * 1048576, 0.75);
    hsize_t blcok_size = 2048 * 1024;
    fapl.setMetaBlockSize(blcok_size);
    fapl.setSieveBufSize(256 * 1024);
    fapl.setDriver(H5FD_SEC2, nullptr);
    H5::H5File h5_file(file_path, H5F_ACC_RDONLY | H5F_ACC_SWMR_READ, H5::FileCreatPropList::DEFAULT, fapl);
    H5::Group Raw_Data = h5_file.openGroup("/Raw_data");
    hsize_t numObjs = Raw_Data.getNumObjs();
    std::unordered_map<std::string , std::vector<float>> data;
    data.reserve(numObjs);
    for (hsize_t i = 0; i < numObjs; i++) {
        std::string read_id = Raw_Data.getObjnameByIdx(i);;
        H5::DataSet dataset = Raw_Data.openDataSet(read_id);
        H5::DataSpace space = dataset.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims, nullptr);
        std::vector<float> signal(dims[0]);
        dataset.read(signal.data(), H5::PredType::NATIVE_FLOAT);
        data[read_id] = std::move(signal);
    }
    auto st2 = std::chrono::high_resolution_clock::now();
    std::ostringstream oss;
    boost::archive::binary_oarchive oa(oss);
    oa << data;
    std::string serialized_data = oss.str();
    // remove possibly existing shared memory
    bip::shared_memory_object::remove(shared_memory_name.c_str());
    bip::managed_shared_memory segment(bip::create_only, shared_memory_name.c_str(), serialized_data.size() + 1024);
    // 存储数据到共享内存
    char* shm_data = segment.construct<char>("SerializedData")[serialized_data.size()]();
    std::memcpy(shm_data, serialized_data.data(), serialized_data.size());
    auto ed = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(st2 - st1);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st2);
//    spdlog::info("write file {} finished, write data size: {} MB, time cost (milliseconds):"
//                 " read data {} , serialization and write to shared mem {}",
//                 shared_memory_name, serialized_data.size() / 1024 / 1024,
//                 duration.count(), duration2.count());
    return 0;
}
