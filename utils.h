//
// Created by yaohc on 2024/12/19.
//

#ifndef STURGEON_UTILS_H
#define STURGEON_UTILS_H
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <hdf5.h>

#include <chrono>
#include <filesystem>
#include <queue>

namespace fs = std::filesystem;


namespace Sturgeon
{
// Recursive function to iterate over the HDF5 groups and datasets
hsize_t iterateHDF5(const H5::Group &group, const fs::path &path = "");

std::vector<fs::path> get_h5_files(const fs::path& path);

int32_t compute_trim_start(const torch::Tensor & signal,
                           int32_t window_size=40,
                           float threshold=2.4,
                           int32_t min_trim = 10,
                           int32_t min_elements=3,
                           int32_t max_samples=8000,
                           float max_trim = 0.3);

std::pair<float, float> get_norm_factors(const torch::Tensor & signal, std::string method = "mad");

/*
 * Thread-safe lock-based queue
 */
template <typename T>
class ThreadSafeQueue {
    std::queue<T> queue_;
    size_t maxSize_;
    std::mutex mutex_;
    std::condition_variable condition_;
public:
    explicit ThreadSafeQueue(size_t maxSize) : maxSize_(maxSize) {}
    ThreadSafeQueue() : maxSize_(5) {}

    void push(const T& item) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return queue_.size() < maxSize_; });
            queue_.push(item);
            lock.unlock();
        }
        condition_.notify_all();
    }

    void push(const T&& item) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return queue_.size() < maxSize_; });
            queue_.push(std::move(item));
            lock.unlock();
        }
        condition_.notify_all();
    }

    // template method to construct elements in place
    template <class... Args>
    decltype(auto) emplace(Args&&... args) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return queue_.size() < maxSize_; });
        auto result = queue_.emplace(std::forward<Args>(args)...);
        lock.unlock();
        condition_.notify_all();
        return result;
    }

    void pop() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return !queue_.empty(); });
            queue_.pop();
            lock.unlock();
        }
        condition_.notify_one();
    }

    const T& front() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        return queue_.front();
    }

    T pop_and_get() {
        T result;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return !queue_.empty(); });
            result = std::move(queue_.front());
            queue_.pop();
//            lock.unlock();
        }
        condition_.notify_all();
        return result;
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    size_t maxSize () {
        return maxSize_;
    }
};

}


#endif //STURGEON_UTILS_H
