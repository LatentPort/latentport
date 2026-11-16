#pragma once
#include <algorithm>

// Allocation size and absolute RoPE coordinates are independent limits.
struct lp_context_policy {
    int position_policy = 0; // 0=original, 1=static YaRN factor 4
    int physical_limit = 262144;
    bool valid() const {
        return (position_policy == 0 || position_policy == 1) &&
               physical_limit >= 32 && physical_limit <= 262144;
    }
    int logical_limit() const { return position_policy == 1 ? 1048576 : 262144; }
    int remaining(int clock, int physical, int capacity) const {
        return std::max(0, std::min({logical_limit() - clock,
                                   physical_limit - physical, capacity - physical}));
    }
};
