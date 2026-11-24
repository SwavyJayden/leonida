#pragma once

#include <cstddef>
#include <cstdint>

namespace rt::xmem {

// First touch from a CPU write that is being emulated. Queries the faulting subpage and commits its
// XMem backing when the view can take a write or has a guard to deliver.
bool commit_physical_page_for_write(const void* address);
bool commit_physical_range(const void* address, std::size_t size);
bool commit_physical_pages(const std::uint64_t* page_array, std::size_t count);

}
