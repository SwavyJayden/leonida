#include "../src/command_list.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {

std::atomic<bool> fail_next_command_allocation = false;
std::atomic<bool> fault_consumed = false;

// The command vector allocates at least sizeof(Command) bytes. Filtering by size keeps the fault away from
// smaller standard-library bookkeeping allocations, including MSVC Debug iterator proxies.
void* allocate(std::size_t size) {
    if (size >= sizeof(leonida::Command) && fail_next_command_allocation.exchange(false)) {
        fault_consumed.store(true);
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}

bool faulted() {
    const bool consumed = fault_consumed.exchange(false);
    fail_next_command_allocation.store(false);
    return consumed;
}

}  // namespace

void* operator new(std::size_t size) { return allocate(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

#define CHECK(condition) do {                                               \
    if (!(condition)) {                                                     \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);      \
        return 1;                                                           \
    }                                                                       \
} while (0)

int main() {
    using namespace leonida;
    const BeginQuery begin{70, d3d12_query::occlusion, 1};
    const EndQuery end{70, d3d12_query::occlusion, 1};

    // The first command allocation fails after the active-query entry has been created.
    CommandList first;
    fail_next_command_allocation.store(true);
    bool threw = false;
    try { first.begin_query(begin); } catch (const std::bad_alloc&) { threw = true; }
    CHECK(faulted() && threw);
    CHECK(first.commands().empty() && !first.queries_open());
    CHECK(first.begin_query(begin));
    CHECK(first.commands().size() == 1 && first.queries_open());
    CHECK(first.end_query(end));
    CHECK(first.commands().size() == 2 && !first.queries_open());

    // Growth of an existing command vector fails with a previous timestamp command still recorded.
    CommandList later;
    CHECK(later.end_query({70, d3d12_query::timestamp, 2}));
    fail_next_command_allocation.store(true);
    threw = false;
    try { later.begin_query(begin); } catch (const std::bad_alloc&) { threw = true; }
    CHECK(faulted() && threw);
    CHECK(later.commands().size() == 1 && !later.queries_open());
    CHECK(std::get<EndQuery>(later.commands()[0]).type == d3d12_query::timestamp);
    CHECK(later.begin_query(begin));
    CHECK(later.commands().size() == 2 && later.queries_open());
    CHECK(later.end_query(end));
    CHECK(later.commands().size() == 3 && !later.queries_open());

    // A failed EndQuery must leave the BeginQuery command and active state available for retry.
    CommandList ending;
    CHECK(ending.begin_query(begin));
    fail_next_command_allocation.store(true);
    threw = false;
    try { ending.end_query(end); } catch (const std::bad_alloc&) { threw = true; }
    CHECK(faulted() && threw);
    CHECK(ending.commands().size() == 1 && ending.queries_open());
    CHECK(std::holds_alternative<BeginQuery>(ending.commands()[0]));
    CHECK(ending.end_query(end));
    CHECK(ending.commands().size() == 2 && !ending.queries_open());

    // Timestamp writes have no active-query entry, and ordinary begin/end pairs remain balanced.
    CommandList normal;
    CHECK(normal.end_query({70, d3d12_query::timestamp, 3}));
    CHECK(normal.end_query({70, d3d12_query::timestamp, 3}));
    CHECK(normal.begin_query(begin) && normal.queries_open());
    CHECK(normal.end_query(end) && !normal.queries_open());
    CHECK(normal.commands().size() == 4);
    CHECK(normal.close() && !normal.queries_open());
    return 0;
}
