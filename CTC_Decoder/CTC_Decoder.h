//
// Created by yaohc on 2024/12/23.
//

#ifndef STURGEON_CTC_DECODER_H
#define STURGEON_CTC_DECODER_H
#include "../DataLoader/DataLoader.h"
#include <vector>
#include <unordered_map>
#include <torch/torch.h>
#include <algorithm>
#include <omp.h>
#include <type_traits>

namespace Sturgeon {

// Template function to calculate safe log-sum-exp for any floating point type
    template<typename T>
    T log_add(T log_a, T log_b) {
        // Ensure that log_a and log_b are finite
        static_assert(std::is_floating_point<T>::value, "Template argument must be a floating-point type");

        // Handle special cases like negative infinity (log(0))
        if (log_a == -std::numeric_limits<T>::infinity()) return log_b;
        if (log_b == -std::numeric_limits<T>::infinity()) return log_a;

        // Find the maximum value between log_a and log_b to avoid overflow
        T max_log = std::max(log_a, log_b);

        // Return max_log + log(1 + exp(-|log_a - log_b|))
        return max_log + std::log1p(std::exp(-std::fabs(log_a - log_b)));
    }

    struct PrefixHash {
        /*
         * implement a hash for std::vector<int>
         * */
        size_t operator() (const std::vector<int>& prefix) const {
            size_t hash_code = 0;
            // here we use KB&DR hash code
            for (int id : prefix) {
                hash_code = id + 31 * hash_code;
            }
            return hash_code;
        }
    };


    uint8_t get_qual(float x);

    struct Decode_Result {
        torch::Tensor seqs;
        torch::Tensor moves;
        torch::Tensor quals;

        Decode_Result slice(int64_t  i, int64_t j) {
            auto seqs_ = seqs.slice(0, i, j);
            auto moves_ = moves.slice(0, i, j);
            auto quals_ = quals.slice(0, i, j);
            return Decode_Result{seqs_, moves_, quals_};
        }
    };

    Decode_Result ctc_greedy_decode(torch::Tensor & logp);


    void ctc_decode_worker(Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch>  & decode_queue,
                           ThreadSafeQueue<std::pair<std::vector<Chunk>, Decode_Result>> & stitch_queue
    );

    struct PrefixScore {
        float s = -std::numeric_limits<float>::max();       // blank ending score
        float ns = -std::numeric_limits<float>::max();      // none blank ending score
        float v_s = -std::numeric_limits<float>::max();
        float v_ns = -std::numeric_limits<float>::max();
        float curr_token_prob = -std::numeric_limits<float>::max();
        std::vector<int> times_s;
        std::vector<int> times_ns;
        float score() const {
            return Sturgeon::log_add(s, ns);
        }
        float viterbi_score() const {
            return v_s > v_ns ? v_s : v_ns;
        }
        const std::vector<int>& times() const {
            return v_s > v_ns ? times_s : times_ns;
        }
    };

    static bool PrefixScoreCompare(
            const std::pair<std::vector<int>, PrefixScore> & a,
            const std::pair<std::vector<int>, PrefixScore> & b);


    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> get_empty_hypo();

    Decode_Result ctc_prefix_beam_Search(const torch::Tensor & logp, size_t beam_size = 5, size_t num_threads = 8);


    class CTC_Prefix_BeamSearch {
    public:
        CTC_Prefix_BeamSearch() {
            std::vector<int> empty;
            PrefixScore empty_score;
            empty_score.s = 0;
            empty_score.ns = -std::numeric_limits<float>::max();
            curr_hypo_[empty] = empty_score;
        }
        void search(const torch::Tensor & logp, size_t beam_size = 5) ;

        void display_hypo() const;
        void clear();
    private:
        std::vector<std::vector<int>> batch_hypo;
        std::vector<std::vector<int>> hypotheses_;
        std::unordered_map<std::vector<int>, Sturgeon::PrefixScore, Sturgeon::PrefixHash> curr_hypo_;
        std::vector<std::vector<int>> times_;
    };
}

#endif //STURGEON_CTC_DECODER_H

