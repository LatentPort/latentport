#include "context.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    lp_context_policy original;
    lp_context_policy yarn{1, 262144};
    assert(original.valid() && yarn.valid());
    assert(original.remaining(262144, 3, 65536) == 0);
    assert(yarn.remaining(262144, 3, 65536) == 65533);
    assert(yarn.remaining(1000000, 100000, 262144) == 48576);
    assert(yarn.remaining(1048576, 3, 65536) == 0);
    assert(yarn.remaining(1000000, 262144, 262144) == 0);
    assert(yarn.remaining(1000000, 65536, 65536) == 0);
    assert((lp_context_policy{1, 4096}.remaining(900000, 4090, 8192) == 6));
    assert(!(lp_context_policy{2, 262144}.valid()));
    assert(!(lp_context_policy{1, 1048576}.valid()));
}
