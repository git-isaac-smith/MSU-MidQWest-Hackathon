#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "pymatching/sparse_blossom/matcher/mwpm.h"

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
struct WeightSlots {
    pm::weight_int *first, *second;
    void set(pm::weight_int value) {
        *first = value;
        if (second) *second = value;
    }
};

// Positive-weight, single-observable path of PyMatching 2.4.0's mwpm_decoding.cc.
// Preserve event insertion, notification processing, and blossom extraction order.
uint8_t solve(pm::Mwpm& matcher, const std::vector<int>& events) {
    auto& flooder = matcher.flooder;
    flooder.queue.cur_time = 0;
    for (int i : events) matcher.create_detection_event(&flooder.graph.nodes[i]);
    while (true) {
        auto event = flooder.run_until_next_mwpm_notification();
        if (event.event_type == pm::NO_EVENT) break;
        matcher.process_event(event);
    }
    if (matcher.node_arena.allocated.size() != matcher.node_arena.available.size())
        throw std::runtime_error("No perfect matching for syndrome");
    pm::obs_int mask = 0;
    for (int i : events) {
        auto& node = flooder.graph.nodes[i];
        if (node.region_that_arrived)
            mask ^= matcher.shatter_blossom_and_extract_matches(node.region_that_arrived_top).obs_mask;
    }
    return static_cast<uint8_t>(mask);
}
}

EXPORT const char* score(int n_nodes, int n_edges, const int32_t* ends, const double* weights,
    const uint8_t* observables, int n_heralds, const int32_t* offsets, const int32_t* conditioned,
    int n_shots, const uint8_t* herald, const uint8_t* syndrome, uint8_t* predictions) {
    static thread_local std::string error;
    try {
        pm::MatchingGraph graph(n_nodes, 1);
        // UserGraph iterates internal edges in insertion order. It appends boundary
        // edges afterwards in node order; each boundary becomes neighbor zero.
        std::vector<int> boundary;
        for (int e = 0; e < n_edges; ++e) {
            if (ends[2*e+1] < 0) { boundary.push_back(e); continue; }
            std::vector<size_t> obs;
            if (observables[e]) obs.push_back(0);
            graph.add_edge(ends[2*e], ends[2*e+1], 0, obs, {});
        }
        std::sort(boundary.begin(), boundary.end(), [&](int a, int b){return ends[2*a] < ends[2*b];});
        for (int e : boundary) {
            std::vector<size_t> obs;
            if (observables[e]) obs.push_back(0);
            graph.add_boundary_edge(ends[2*e], 0, obs, {});
        }
        pm::Mwpm matcher{pm::GraphFlooder(std::move(graph))};
        auto& nodes = matcher.flooder.graph.nodes;
        std::vector<WeightSlots> slots;
        std::vector<int> descending(n_edges);
        std::iota(descending.begin(), descending.end(), 0);
        std::sort(descending.begin(), descending.end(), [&](int a, int b){return weights[a] > weights[b];});
        int fractional = 0;
        for (int e = 0; e < n_edges; ++e) {
            auto& u = nodes[ends[2*e]];
            auto v = ends[2*e+1] < 0 ? nullptr : &nodes[ends[2*e+1]];
            slots.push_back({&u.neighbor_weights.at(u.index_of_neighbor(v)),
                v ? &v->neighbor_weights.at(v->index_of_neighbor(&u)) : nullptr});
            fractional += weights[e] != std::round(weights[e]);
        }
        std::vector<uint8_t> erased(n_edges);
        std::vector<pm::weight_int> original(n_edges);
        std::vector<int> touched, events;
        double last_scale = -1;
        for (int shot = 0; shot < n_shots; ++shot) {
            touched.clear();
            int remaining_fractional = fractional;
            for (int h = 0; h < n_heralds; ++h) {
                if (!herald[size_t(shot)*n_heralds+h]) continue;
                for (int j = offsets[h]; j < offsets[h+1]; ++j) {
                    int e = conditioned[j];
                    if (erased[e]) continue;
                    erased[e] = 1;
                    touched.push_back(e);
                    remaining_fractional -= weights[e] != std::round(weights[e]);
                }
            }
            double scale = 1;
            if (remaining_fractional) {
                auto greatest = std::find_if(descending.begin(), descending.end(), [&](int e){return !erased[e];});
                scale = double((1u << 24) - 1) / weights[*greatest];
            }
            if (scale != last_scale) {
                for (int e = 0; e < n_edges; ++e) {
                    original[e] = 2 * static_cast<pm::weight_int>(std::round(weights[e] * scale));
                    slots[e].set(original[e]);
                }
                last_scale = scale;
            }
            for (int e : touched) slots[e].set(0);
            events.clear();
            for (int i = 0; i < n_nodes; ++i)
                if (syndrome[size_t(shot)*n_nodes+i]) events.push_back(i);
            predictions[shot] = solve(matcher, events);
            for (int e : touched) { slots[e].set(original[e]); erased[e] = 0; }
        }
        return nullptr;
    } catch (const std::exception& e) {
        error = e.what();
        return error.c_str();
    }
}
