#include "neat/innovation.hpp"

namespace neat {

InnovationTracker::InnovationTracker(uint32_t start_counter)
    : counter_(start_counter) {}

uint32_t InnovationTracker::get_or_assign(uint32_t from, uint32_t to) {
    uint64_t key = (static_cast<uint64_t>(from) << 32) | to;
    auto it = generation_cache_.find(key);
    if (it != generation_cache_.end()) {
        return it->second;
    }
    uint32_t innovation = counter_++;
    generation_cache_[key] = innovation;
    return innovation;
}

void InnovationTracker::next_generation() {
    generation_cache_.clear();
}

uint32_t InnovationTracker::current_counter() const {
    return counter_;
}

} // namespace neat
