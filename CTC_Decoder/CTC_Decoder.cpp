#include "CTC_Decoder.h"

uint8_t Sturgeon::get_qual(float x) {
    x = x < 1e-7 ? 1e-7 : x;
    x = x > (1.0 - 1e-7) ? (1.0 - 1e-7) : x;
    return static_cast<uint8_t>(-10 * std::log10(1 - x) + 33);
}

Sturgeon::Decode_Result Sturgeon::ctc_greedy_decode(torch::Tensor &logp) {
    int64_t T = logp.size(0), N = logp.size(1), C = logp.size(2);
    torch::Tensor soft_inputs = torch::exp(logp).permute({1, 0, 2}).contiguous();
    torch::Tensor logits = soft_inputs.argmax(2).contiguous();
    auto soft_inputs_ptr = soft_inputs.accessor<float, 3>(); // T x N x C
    auto logits_ptr = logits.accessor<int64_t, 2>(); // N x T


    auto seqs = torch::zeros({N, T}, torch::kUInt8);
    auto moves = torch::zeros({N, T}, torch::kUInt8);
    auto quals = torch::zeros({N, T}, torch::kUInt8);
    auto seqs_ptr = seqs.accessor<uint8_t, 2>();
    auto moves_ptr = moves.accessor<uint8_t, 2>();
    auto quals_ptr = quals.accessor<uint8_t, 2>();

    for (int i = 0; i < N; i++) {
        if (logits_ptr[i][0] != 0) {
            seqs[i][0] = logits_ptr[i][0];
            moves[i][0] = 1;
            int64_t k = logits_ptr[i][0];
            quals_ptr[i][0] = get_qual(soft_inputs_ptr[i][0][k]);
        }
        for (int j = 1; j < T; j++) {
            if (logits_ptr[i][j] != 0 && \
                        logits_ptr[i][j - 1] != logits_ptr[i][j]) {
                seqs_ptr[i][j] = logits_ptr[i][j];
                moves_ptr[i][j] = 1;
                int64_t k = logits_ptr[i][j];
                quals_ptr[i][j] = get_qual(soft_inputs_ptr[i][j][k]);
            }
        }
    }

    return {seqs, moves, quals};
}


void Sturgeon::ctc_decode_worker(Sturgeon::ThreadSafeQueue<Sturgeon::Sig_Batch>  &decode_queue,
                                 ThreadSafeQueue<std::pair<std::vector<Chunk>, Decode_Result>> &stitch_queue) {
    while(true) {
        auto sig_batch = decode_queue.pop_and_get();
//        auto st = std::chrono::high_resolution_clock::now();
        if (sig_batch.empty()) break;
        torch::Tensor logp = torch::cat(sig_batch.results, 1);
        Sturgeon::Decode_Result res = ctc_greedy_decode(logp);
        int64_t i = 0, j = 0;
        int64_t n = sig_batch.chunks.size();
        while(j < n) {
            while(j < n && sig_batch.chunks[j].meta->read_id == sig_batch.chunks[i].meta->read_id) j += 1;
            auto item1 = std::vector<Chunk> (sig_batch.chunks.begin() + i, sig_batch.chunks.begin() + j);
            auto item2 = res.slice(i, j);
            // push chunks per read to stitch results
            stitch_queue.emplace(std::move(item1), item2);
            i = j;
        }
//        auto ed = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st);
//        spdlog::info("CTC decode time: {} ms, samples: {}", duration.count(), sig_batch.chunks.size());
    }
    stitch_queue.emplace(std::vector<Chunk>(), Decode_Result());
    spdlog::info("CTC decode finished");
}



// search for a batch of CTC inputs
void Sturgeon::CTC_Prefix_BeamSearch::search(const torch::Tensor &logp, size_t beam_size) {

    if (logp.size(0) == 0) return;
    size_t T = logp.size(0), N = logp.size(1), C = logp.size(2);
    torch::Tensor logp_ = logp.permute({1, 0, 2}).contiguous();
    for (size_t n = 0; n < N; ++n){
        const float *logp_t = logp_[n][0].data_ptr<float>(); // use raw pointer to accelerate element access speed
        for (size_t t = 0; t < T; ++t) {
            std::unordered_map<std::vector<int>, Sturgeon::PrefixScore, Sturgeon::PrefixHash> next_hyps;
            // 1. first beam prune, only select topk candidates, pass this part and implement it in the furture
            // 2. Toke passing
            for (size_t id = 0; id < C; id++) {
                auto prob = logp_t[t * C + id];
                for (const auto &it: curr_hypo_) {
                    const std::vector<int> &prefix = it.first;
                    const Sturgeon::PrefixScore &prefix_score = it.second;
                    if (id == 0) {
                        // Case 0: *a + ε => *a
                        PrefixScore &next_score = next_hyps[prefix];
                        next_score.s = Sturgeon::log_add(next_score.s, prefix_score.score() + prob);
                    } else if (!prefix.empty() && id == prefix.back()) {
                        // Case 1: *a + a => *a
                        PrefixScore &next_score = next_hyps[prefix];
                        next_score.ns = Sturgeon::log_add(next_score.ns, prefix_score.ns + prob);
                        // Case 2: *aε + a => *aa
                        std::vector<int> new_prefix(prefix);
                        new_prefix.emplace_back(id);
                        PrefixScore &next_score2 = next_hyps[new_prefix];
                        next_score2.ns = Sturgeon::log_add(next_score2.ns, prefix_score.s + prob);
                    } else {
                        // Case 3: *a + b => *ab, *aε + b => *ab
                        std::vector<int> new_prefix(prefix);
                        new_prefix.emplace_back(id);
                        PrefixScore &next_score = next_hyps[new_prefix];
                        next_score.ns = Sturgeon::log_add(next_score.ns, prefix_score.score() + prob);
                    }
                }
            }
            // 3. beam prune, only keep top n best paths
            std::vector<std::pair<std::vector<int>, Sturgeon::PrefixScore>> arr(next_hyps.begin(), next_hyps.end());
            size_t real_beam_size = std::min(arr.size(), beam_size);
            std::nth_element(
                    arr.begin(), arr.begin() + real_beam_size, arr.end(),
                    Sturgeon::PrefixScoreCompare);
            arr.resize(real_beam_size);
            std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

            // 4. update new hypotheses
            curr_hypo_.clear();
            hypotheses_.clear();
            for (auto &item: arr) {
                curr_hypo_[item.first] = item.second;
                hypotheses_.emplace_back(item.first);
            }
        }
        batch_hypo.push_back(hypotheses_.front());
        clear();
    }
}

void Sturgeon::CTC_Prefix_BeamSearch::display_hypo() const {
    for (const std::vector<int> & vec : batch_hypo) {
        std::cout << "hypo: ";
        for (auto &id : vec) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
    }
}

void Sturgeon::CTC_Prefix_BeamSearch::clear() {
    hypotheses_.clear();
    curr_hypo_.clear();
    PrefixScore empty_score;
    empty_score.s = 0;
    empty_score.ns = -std::numeric_limits<float>::max();
    curr_hypo_[std::vector<int>()] = empty_score;
}

bool Sturgeon::PrefixScoreCompare(const std::pair<std::vector<int>, PrefixScore> &a,
                                  const std::pair<std::vector<int>, PrefixScore> &b) {
    return a.second.score() > b.second.score();
}


