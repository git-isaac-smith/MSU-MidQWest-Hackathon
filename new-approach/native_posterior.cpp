#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define POSTERIOR_API extern "C" __declspec(dllexport)
#else
#define POSTERIOR_API extern "C"
#endif

namespace {

constexpr std::size_t syndrome_count = 1 << 12;
constexpr std::size_t state_count = 2 * syndrome_count;
constexpr std::size_t round_count = 5;

struct alignas(32) Message {
    std::array<double, state_count> weights;
};

struct Fault {
    std::size_t shift;
    std::size_t high_bit;
    double probability;
};

#if defined(__AVX2__)

template<int lanes>
__m256d lane_xor(__m256d values) {
    constexpr int permutation = (0 ^ lanes) | ((1 ^ lanes) << 2)
                                | ((2 ^ lanes) << 4) | ((3 ^ lanes) << 6);
    if constexpr (lanes == 0) {
        return values;
    } else {
        return _mm256_permute4x64_pd(values, permutation);
    }
}

template<int lanes>
void transition_pairs(double* values, const Fault& fault) {
    const auto p = _mm256_set1_pd(fault.probability);
    const auto q = _mm256_set1_pd(1 - fault.probability);
    const auto block_xor = fault.shift & ~std::size_t{3};
    for (std::size_t start = 0; start < state_count; start += 2 * fault.high_bit) {
        for (std::size_t i = start; i < start + fault.high_bit; i += 4) {
            const auto j = i ^ block_xor;
            const auto left = _mm256_load_pd(values + i);
            const auto right = _mm256_load_pd(values + j);
            _mm256_store_pd(values + i, _mm256_add_pd(_mm256_mul_pd(left, q),
                                                     _mm256_mul_pd(lane_xor<lanes>(right), p)));
            _mm256_store_pd(values + j, _mm256_add_pd(_mm256_mul_pd(right, q),
                                                     _mm256_mul_pd(lane_xor<lanes>(left), p)));
        }
    }
}

template<int lanes>
void transition_lanes(double* values, double probability) {
    const auto p = _mm256_set1_pd(probability);
    const auto q = _mm256_set1_pd(1 - probability);
    for (std::size_t i = 0; i < state_count; i += 4) {
        const auto value = _mm256_load_pd(values + i);
        _mm256_store_pd(values + i, _mm256_add_pd(_mm256_mul_pd(value, q),
                                                 _mm256_mul_pd(lane_xor<lanes>(value), p)));
    }
}

__m256i lane_indices(std::size_t mask) {
    return _mm256_xor_si256(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                            _mm256_set1_epi32(static_cast<int>((mask & 3) * 2)));
}

__m256d xor_load(const double* values, std::size_t index, __m256i indices) {
    const auto loaded = _mm256_castpd_si256(_mm256_load_pd(values + index));
    return _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(loaded, indices));
}

#endif

void transition(Message& message, const Fault& fault) {
    double* values = message.weights.data();
#if defined(__AVX2__)
    if (fault.high_bit >= 4) {
        switch (fault.shift & 3) {
            case 0: return transition_pairs<0>(values, fault);
            case 1: return transition_pairs<1>(values, fault);
            case 2: return transition_pairs<2>(values, fault);
            case 3: return transition_pairs<3>(values, fault);
        }
    }
    switch (fault.shift) {
        case 1: return transition_lanes<1>(values, fault.probability);
        case 2: return transition_lanes<2>(values, fault.probability);
        case 3: return transition_lanes<3>(values, fault.probability);
    }
#endif
    const auto p = fault.probability;
    const auto q = 1 - p;
    for (std::size_t start = 0; start < state_count; start += 2 * fault.high_bit) {
        for (std::size_t i = start; i < start + fault.high_bit; ++i) {
            const auto j = i ^ fault.shift;
            const auto left = values[i];
            const auto right = values[j];
            values[i] = q * left + p * right;
            values[j] = q * right + p * left;
        }
    }
}

struct PrefixEntry {
    std::size_t key = syndrome_count * syndrome_count;
    std::unique_ptr<Message> message;
};

class Decoder {
public:
    Decoder(const double* prior, const double* readout, const std::uint16_t* shifts,
            const double* probabilities, std::size_t fault_count, std::size_t prefix_capacity)
        : prefix_cache_(prefix_capacity) {
        for (std::size_t i = 0; i < state_count; ++i) {
            prior_.weights[i] = prior[i];
        }
        for (std::size_t i = 0; i < syndrome_count; ++i) {
            readout_[i] = readout[i];
        }
        for (std::size_t i = 0; i < fault_count; ++i) {
            const std::size_t shift = shifts[i];
            if (shift == 0 || shift >= state_count) {
                throw std::invalid_argument("invalid fault shift");
            }
            std::size_t high_bit = 1;
            while (2 * high_bit <= shift) {
                high_bit *= 2;
            }
            faults_.push_back({shift, high_bit, probabilities[i]});
        }
    }

    bool predict(const std::uint16_t* observations, std::size_t terminal, double& result) {
        // At round three, the backward message is a two-step update from the
        // final syndrome. Translation by that syndrome puts it in the same
        // coordinates as the one-observation message from the initial state.
        const auto& forward = prefix_message(observations[0], observations[1]);
        const auto& backward = single_message(observations[3] ^ terminal);
        double zero;
        double one;
        logical_masses(forward, backward, observations[2], terminal, zero, one);
        const double total = zero + one;
        if (total == 0) {
            return false;
        }
        result = one / total;
        return true;
    }

private:
    void update(const Message& source, std::size_t observed, Message& target) const {
        // The wrapper uses the reference implementation's log-space path
        // whenever a possible history could underflow in double precision.
        // Otherwise we can propagate the nonnegative joint masses directly.
        const auto& in = source.weights;
        auto& out = target.weights;
#if defined(__AVX2__)
        const auto lanes = lane_indices(observed);
        const auto block = observed & ~std::size_t{3};
        for (std::size_t i = 0; i < syndrome_count; i += 4) {
            const auto emission = xor_load(readout_.data(), i ^ block, lanes);
            _mm256_store_pd(out.data() + i, _mm256_mul_pd(emission, _mm256_load_pd(in.data() + i)));
            _mm256_store_pd(out.data() + syndrome_count + i,
                            _mm256_mul_pd(emission, _mm256_load_pd(in.data() + syndrome_count + i)));
        }
#else
        for (std::size_t i = 0; i < syndrome_count; ++i) {
            const double emission = readout_[i ^ observed];
            out[i] = emission * in[i];
            out[i + syndrome_count] = emission * in[i + syndrome_count];
        }
#endif
        for (const auto& fault : faults_) {
            transition(target, fault);
        }
    }

    const Message& single_message(std::size_t observed) {
        auto& entry = single_cache_[observed];
        if (!entry) {
            entry.reset(new Message);
            update(prior_, observed, *entry);
        }
        return *entry;
    }

    const Message& prefix_message(std::size_t first, std::size_t second) {
        const std::size_t key = first * syndrome_count + second;
        // Mix the two words so common small syndrome masks do not all map to
        // the same slot when the cache size is a power of two.
        const std::uint64_t hash = (key ^ (key >> 12)) * UINT64_C(0x9e3779b97f4a7c15);
        auto& entry = prefix_cache_[(hash >> 32) % prefix_cache_.size()];
        if (entry.key != key) {
            if (!entry.message) {
                entry.message.reset(new Message);
            }
            update(single_message(first), second, *entry.message);
            entry.key = key;
        }
        return *entry.message;
    }

    void logical_masses(const Message& forward, const Message& backward,
                        std::size_t middle, std::size_t terminal, double& zero, double& one) const {
        const auto* f0 = forward.weights.data();
        const auto* f1 = f0 + syndrome_count;
        const auto* b0 = backward.weights.data();
        const auto* b1 = b0 + syndrome_count;
#if defined(__AVX2__)
        const auto middle_lanes = lane_indices(middle);
        const auto terminal_lanes = lane_indices(terminal);
        const auto middle_block = middle & ~std::size_t{3};
        const auto terminal_block = terminal & ~std::size_t{3};
        auto sum_zero = _mm256_setzero_pd();
        auto sum_one = _mm256_setzero_pd();
        for (std::size_t i = 0; i < syndrome_count; i += 4) {
            const auto emission = xor_load(readout_.data(), i ^ middle_block, middle_lanes);
            const auto left0 = _mm256_mul_pd(emission, _mm256_load_pd(f0 + i));
            const auto left1 = _mm256_mul_pd(emission, _mm256_load_pd(f1 + i));
            const auto right0 = xor_load(b0, i ^ terminal_block, terminal_lanes);
            const auto right1 = xor_load(b1, i ^ terminal_block, terminal_lanes);
            sum_zero = _mm256_add_pd(sum_zero, _mm256_add_pd(_mm256_mul_pd(left0, right0), _mm256_mul_pd(left1, right1)));
            sum_one = _mm256_add_pd(sum_one, _mm256_add_pd(_mm256_mul_pd(left0, right1), _mm256_mul_pd(left1, right0)));
        }
        alignas(32) double lanes_zero[4];
        alignas(32) double lanes_one[4];
        _mm256_store_pd(lanes_zero, sum_zero);
        _mm256_store_pd(lanes_one, sum_one);
        zero = (lanes_zero[0] + lanes_zero[1]) + (lanes_zero[2] + lanes_zero[3]);
        one = (lanes_one[0] + lanes_one[1]) + (lanes_one[2] + lanes_one[3]);
#else
        zero = 0;
        one = 0;
        for (std::size_t i = 0; i < syndrome_count; ++i) {
            const std::size_t j = i ^ terminal;
            const double emission = readout_[i ^ middle];
            const double left0 = emission * f0[i];
            const double left1 = emission * f1[i];
            zero += left0 * b0[j] + left1 * b1[j];
            one += left0 * b1[j] + left1 * b0[j];
        }
#endif
    }

    Message prior_;
    alignas(32) std::array<double, syndrome_count> readout_;
    std::vector<Fault> faults_;
    std::array<std::unique_ptr<Message>, syndrome_count> single_cache_;
    std::vector<PrefixEntry> prefix_cache_;
};

}  // namespace

POSTERIOR_API void* posterior_create(const double* prior, const double* readout,
                                    const std::uint16_t* shifts, const double* probabilities,
                                    std::size_t fault_count, std::size_t prefix_capacity) noexcept {
    if (prefix_capacity == 0) {
        return nullptr;
    }
    try {
        return new Decoder(prior, readout, shifts, probabilities, fault_count, prefix_capacity);
    } catch (...) {
        return nullptr;
    }
}

POSTERIOR_API void posterior_destroy(void* handle) noexcept {
    delete static_cast<Decoder*>(handle);
}

POSTERIOR_API int posterior_predict(void* handle, const std::uint16_t* observed,
                                    const std::uint16_t* terminal, std::size_t count,
                                    double* results) noexcept {
    auto& decoder = *static_cast<Decoder*>(handle);
    try {
        for (std::size_t i = 0; i < count; ++i) {
            if (!decoder.predict(observed + i * round_count, terminal[i], results[i])) {
                return 1;
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}
