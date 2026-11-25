// Buffer descriptors that name memory nothing backs become zero-size (docs/d3d12_vulkan.md section 3.3, "Title-chosen
// addresses"): a shader adds heap_base to a V#'s base, and heap_base + an address no allocation holds is a GPU MMU
// fault. Each path the layer sees a buffer address on writes such an address as a V# with no base and no records
// (loads return 0, stores are dropped) instead of passing it through:
//   - created views: CreateConstantBufferView and a slot-70 placed buffer view of an unmapped address;
//   - inline V#s: a V# the CPU writes into a bound table of a shader-visible heap itself, rewritten before the
//     submission;
//   - a console descriptor heap (Xbox slot 14) the title writes itself: its GPU copy's slot is null, the title's
//     bytes are left as written;
//   - and the null slots after a resource heap's GPU copy: an index past the heap reads a null descriptor, not the
//     arena's next bytes.
// A data slot (neither a V# nor a T#) of a console heap is null in its GPU copy too.
// Positive controls: an arena buffer's V#, an upload buffer's V# (outside the arena, a registered buffer) and an
// image-shaped slot keep their bytes. Nothing here dispatches through a descriptor, so a regression shows as a
// failed check, never as a GPU fault. No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"
#include "../src/shader_library.h"
#include "../src/subgroup_size.h"
#include "../src/title_page_watch.h"
#include "../src/title_va_alias.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace rt::d3d12 {
bool rdna2_code_walks_bvh(const std::uint8_t* code, std::size_t size);
bool compute_program_walks_bvh_for_test(const leonida::ComputeProgram& program);
bool evaluate_bvh_hold_policy(const char* env_value, std::string_view name, std::string_view sha256,
                              bool walks_bvh, const leonida::SubgroupSizeCaps& subgroup,
                              bool rehydrate_unimplemented, const char** reason = nullptr);
}  // namespace rt::d3d12

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr UINT kDescriptor = 32;
constexpr UINT kViewBytes = 256;

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

leonida::gfx10::BufferWords raw_vsharp(std::uint64_t address) {
    leonida::gfx10::Buffer b{};
    b.address = address;
    b.num_records = kViewBytes;
    b.format = leonida::gfx10::format_from_vulkan(VK_FORMAT_R32_UINT);
    b.oob_select = leonida::gfx10::OobSelect::raw;
    return leonida::gfx10::encode(b);
}

std::uint64_t base_of(const std::uint32_t* w) { return std::uint64_t(w[0]) | (std::uint64_t(w[1] & 0xFFFFu) << 32); }
bool zero_size(const std::uint32_t* w) { return base_of(w) == 0 && w[2] == 0; }

// An address no allocation holds and that is not the title's memory: reserved, then released. Top-down, so it lies
// far past the arena (which is below 4 GiB).
std::uint64_t unmapped_address() {
    void* p = VirtualAlloc(nullptr, 1 << 20, MEM_TOP_DOWN | MEM_RESERVE, PAGE_NOACCESS);
    if (!p) return 0;
    VirtualFree(p, 0, MEM_RELEASE);
    return reinterpret_cast<std::uint64_t>(p) + 0x1000;
}

std::vector<std::uint8_t> serialize(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto fn = d3d12 ? reinterpret_cast<SerializeFn>(
                                reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")))
                          : nullptr;
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (fn && SUCCEEDED(fn(&desc, &blob, nullptr)) && blob) {
        const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(p, p + blob->GetBufferSize());
        blob->Release();
    }
    return out;
}

struct Buffers {
    Ref<ID3D12Resource> arena, upload;
    std::uint64_t arena_va = 0, upload_va = 0;
};

int make_buffers(ID3D12Device* device, std::uint64_t arena_size, Buffers& b) {
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const auto desc = buffer_desc(64 * 1024);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b.arena.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                     IID_PPV_ARGS(&b.upload.p))));
    b.arena_va = b.arena->GetGPUVirtualAddress();
    b.upload_va = b.upload->GetGPUVirtualAddress();
    EXPECT(b.arena_va && b.arena_va < arena_size);
    EXPECT(b.upload_va);
    return 0;
}

// Created views: a CBV and a slot-70 buffer SRV of an unmapped address get no base and no records; a CBV of an
// arena buffer and of an upload buffer keep their addresses.
int check_created_views(ID3D12Device* device, const Buffers& b, std::uint64_t unmapped) {
    alignas(32) std::uint32_t slots[4][8] = {};
    const auto handle = [&](int i) { return D3D12_CPU_DESCRIPTOR_HANDLE{reinterpret_cast<SIZE_T>(slots[i])}; };
    const D3D12_CONSTANT_BUFFER_VIEW_DESC nowhere{unmapped, kViewBytes}, arena{b.arena_va, kViewBytes},
        upload{b.upload_va, kViewBytes};
    device->CreateConstantBufferView(&nowhere, handle(0));
    device->CreateConstantBufferView(&arena, handle(1));
    device->CreateConstantBufferView(&upload, handle(2));
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = kViewBytes / 4;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    EXPECT(rt::d3d12::write_placed_buffer_view(device, false, &srv, unmapped, kViewBytes, handle(3)));
    EXPECT(zero_size(slots[0]));
    EXPECT(base_of(slots[1]) == b.arena_va && slots[1][2] != 0);
    EXPECT(base_of(slots[2]) == (b.upload_va & 0xFFFFFFFFFFFFull) && slots[2][2] != 0);
    EXPECT(zero_size(slots[3]));
    std::printf("created views: a CBV and a placed SRV of unmapped 0x%llx are zero-size; arena and upload CBVs kept\n",
                static_cast<unsigned long long>(unmapped));
    return 0;
}

// Inline V#s in a bound table of one of the layer's heaps, written by the CPU itself as the title does: before the
// submission an unmapped base becomes zero-size in place; arena, upload and image-shaped slots keep their bytes.
int check_inline_table(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64& fence_value,
                       const Buffers& b, std::uint64_t unmapped) {
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
    auto* cpu = reinterpret_cast<std::uint8_t*>(heap->GetCPUDescriptorHandleForHeapStart().ptr);
    const auto put = [&](int slot, const leonida::gfx10::BufferWords& w) {
        std::memcpy(cpu + slot * kDescriptor, w.data(), sizeof(w));
    };
    put(0, raw_vsharp(unmapped));
    for (int i = 4; i < 131; ++i) put(i, raw_vsharp(unmapped));
    put(1, raw_vsharp(b.arena_va));
    put(2, raw_vsharp(b.upload_va & 0xFFFFFFFFFFFFull));
    auto image_like = raw_vsharp(unmapped);
    image_like[3] |= 0x8u << 28;  // TYPE: an image
    put(3, image_like);
    std::uint8_t kept[3][kDescriptor];
    for (int i = 0; i < 3; ++i) std::memcpy(kept[i], cpu + (i + 1) * kDescriptor, kDescriptor);

    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 8, 0, 0, 0};
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable = {1, &range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize(root_desc);
    EXPECT(!blob.empty());
    Ref<ID3D12RootSignature> root;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    ID3D12DescriptorHeap* heaps[] = {heap.p};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(root.p);
    list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(SUCCEEDED(queue->Signal(fence, ++fence_value)));
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));

    const auto* slot0 = reinterpret_cast<const std::uint32_t*>(cpu);
    EXPECT(zero_size(slot0));
    for (int i = 0; i < 3; ++i) EXPECT(!std::memcmp(kept[i], cpu + (i + 1) * kDescriptor, kDescriptor));
    std::printf("inline table V#s: unmapped base written zero-size before the submission; arena, upload and "
                "image-shaped slots kept\n");
    return 0;
}

// A console descriptor heap the title fills itself: its GPU copy nulls the V# that names unmapped memory and keeps the
// others; the title's own bytes are left as written. After the copy come null slots, which no other allocation takes.
int check_console_heap(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* fence,
                       UINT64& fence_value, const Buffers& b, std::uint64_t unmapped) {
    constexpr UINT slots = 2048;
    constexpr std::uint64_t bytes = std::uint64_t(slots) * kDescriptor;
    auto* title = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    EXPECT(title);
    // Written before the heap exists: the first refresh copies and fixes every slot (no write tracking needed).
    const auto put = [&](int slot, const leonida::gfx10::BufferWords& w) {
        std::memcpy(title + slot * kDescriptor, w.data(), sizeof(w));
    };
    put(0, raw_vsharp(unmapped));
    for (int i = 4; i < 131; ++i) put(i, raw_vsharp(unmapped));
    put(1, raw_vsharp(b.arena_va));
    put(2, raw_vsharp(b.upload_va & 0xFFFFFFFFFFFFull));
    // Slot 3: data the title keeps in a slot, neither a V# (TYPE 4) nor a T#; read as a V# its words name nothing.
    const std::uint32_t data[8] = {0x12345000u, 0x00007777u, 0x00000400u, 0x40000000u, 1, 2, 3, 4};
    std::memcpy(title + 3 * kDescriptor, data, sizeof(data));
    std::uint8_t before[4][kDescriptor];
    for (int i = 0; i < 4; ++i) std::memcpy(before[i], title + i * kDescriptor, kDescriptor);

    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots,
                                          D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(rt::d3d12::create_console_descriptor_heap(device, &desc, reinterpret_cast<std::uint64_t>(title),
                                                               bytes, IID_PPV_ARGS(&heap.p))));
    D3D12_DESCRIPTOR_HEAP_DESC got{};
    D3D12_GPU_VIRTUAL_ADDRESS shader_va = 0;
    EXPECT(rt::d3d12::descriptor_heap_info(heap.p, got, shader_va));
    std::uint32_t copy[4][kDescriptor / 4];
    EXPECT(rt::d3d12::read_arena(device, shader_va, copy, sizeof(copy)));
    EXPECT(zero_size(copy[0]));
    EXPECT(!std::memcmp(copy[1], before[1], kDescriptor));
    EXPECT(!std::memcmp(copy[2], before[2], kDescriptor));
    EXPECT(!(copy[3][0] | copy[3][1] | copy[3][2] | copy[3][3]));  // the data slot's descriptor half is null
    for (int i = 0; i < 4; ++i) EXPECT(!std::memcmp(title + i * kDescriptor, before[i], kDescriptor));

    // 128 cold descriptors share one absent range. Publishing its alias leaves the title bytes
    // unchanged. Only the bounded round-robin probes should discover it in the first 64 submits.
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    for (unsigned pass = 0; pass < 64; ++pass) {
        if (pass == 1) leonida::title_va_aliases().add(unmapped, kViewBytes, b.arena_va);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(SUCCEEDED(queue->Signal(fence, ++fence_value)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
    }
    unsigned restored = 0;
    for (unsigned slot = 0; slot < 131; ++slot) {
        if (slot > 0 && slot < 4) continue;
        std::uint32_t words[8];
        EXPECT(rt::d3d12::read_arena(device, shader_va + slot * kDescriptor, words, sizeof(words)));
        if (base_of(words) == b.arena_va && words[2]) ++restored;
        else EXPECT(zero_size(words));
    }
    EXPECT(restored == 16);
    std::printf("128 settled zero descriptors, %u restored in 64 submissions\n", restored);
    leonida::title_va_aliases().remove(unmapped, b.arena_va);

    // The null guard: past the heap's last slot, up to kHeapIndexReach from its start, every byte reads 0.
    constexpr std::uint64_t reach = 64ull << 20;
    std::vector<std::uint8_t> tail(1 << 20);
    for (std::uint64_t at = bytes; at < reach; at += tail.size()) {
        const std::size_t n = std::size_t(std::min<std::uint64_t>(tail.size(), reach - at));
        EXPECT(rt::d3d12::read_arena(device, shader_va + at, tail.data(), n));
        for (std::size_t i = 0; i < n; ++i)
            if (tail[i]) {
                std::fprintf(stderr, "guard byte +0x%llx is 0x%02x\n", static_cast<unsigned long long>(at + i), tail[i]);
                return 1;
            }
    }
    // Reserved with the heap: a buffer made now lies outside the copy and its guard.
    Ref<ID3D12Resource> later;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto later_desc = buffer_desc(64 * 1024);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &later_desc,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&later.p))));
    const std::uint64_t later_va = later->GetGPUVirtualAddress();
    EXPECT(later_va + 64 * 1024 <= shader_va || later_va >= shader_va + reach);
    std::printf("console heap: GPU copy at 0x%llx nulls the unmapped V# and a data slot, keeps arena and upload V#s, title bytes "
                "intact; %llu MiB of null slots after it\n",
                static_cast<unsigned long long>(shader_va), static_cast<unsigned long long>(reach >> 20));
    heap.reset();
    VirtualFree(title, 0, MEM_RELEASE);
    return 0;
}

// A bindless-sized heap of the layer's own (64Ki slots): an index past it reads null slots up to the reach.
int check_layer_heap_guard(ID3D12Device* device) {
    constexpr UINT slots = 65536;
    constexpr std::uint64_t bytes = std::uint64_t(slots) * kDescriptor, reach = 64ull << 20;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots,
                                          D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p))));
    const std::uint64_t base = heap->GetGPUDescriptorHandleForHeapStart().ptr;
    std::vector<std::uint8_t> tail(1 << 20);
    for (std::uint64_t at = bytes; at < bytes + reach; at += tail.size()) {
        EXPECT(rt::d3d12::read_arena(device, base + at, tail.data(), tail.size()));
        for (const std::uint8_t v : tail) EXPECT(v == 0);
    }
    Ref<ID3D12Resource> later;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto later_desc = buffer_desc(64 * 1024);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &later_desc,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&later.p))));
    const std::uint64_t later_va = later->GetGPUVirtualAddress();
    EXPECT(later_va + 64 * 1024 <= base || later_va >= base + bytes + reach);
    std::printf("layer heap: %u slots, then %llu MiB of null slots no other allocation takes\n", slots,
                static_cast<unsigned long long>(reach >> 20));
    return 0;
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    const std::size_t at = out.size();
    out.resize(at + 4);
    std::memcpy(out.data() + at, &v, 4);
}

void put_u64_at(std::vector<std::uint8_t>& out, std::size_t at, std::uint64_t v) {
    std::memcpy(out.data() + at, &v, 8);
}

std::vector<std::uint8_t> make_synthetic_fssm(const std::vector<std::uint32_t>& words) {
    constexpr std::size_t kCodeOff = 0x100;
    constexpr std::size_t kCodeStart = kCodeOff + 0x28;
    std::vector<std::uint8_t> blob(kCodeStart, 0);
    std::memcpy(blob.data(), "FSSM", 4);
    put_u64_at(blob, 0xC0, kCodeOff);
    put_u64_at(blob, 0xC8, 0x28 + words.size() * 4);
    put_u64_at(blob, 0xE8, 0xDEADBEEFDEADBEEFull);
    for (const std::uint32_t w : words) put_u32(blob, w);
    return blob;
}

// Blocker 4: RDNA2 MIMG image_bvh(_64)_intersect_ray scanner and LEONIDA_BVH_HOLD policy evaluation.
int check_bvh_hold_and_scanner() {
    const auto scan_words = [](const std::vector<std::uint32_t>& words) {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(words.size() * 4);
        for (const std::uint32_t w : words) put_u32(bytes, w);
        return rt::d3d12::rdna2_code_walks_bvh(bytes.data(), bytes.size());
    };

    // Plain compute shader (s_mov_b32 s0, 1; s_endpgm): no BVH walk.
    EXPECT(!scan_words({0xBE800381u, 0xBF810000u}));
    // GFX10 MIMG image_bvh_intersect_ray (opc 0xE6: hi6=0x3C, bits[24:18]=0x66, bit0=1) + s_endpgm.
    EXPECT(scan_words({0xF1980001u, 0x00010000u, 0xBF810000u}));
    // GFX10 MIMG image_bvh64_intersect_ray (opc 0xE7) with 3 NSA dwords (20-byte instruction) + s_endpgm.
    EXPECT(scan_words({0xBE8003FFu, 0x12345678u, 0xF19C0007u, 0x00010000u, 0x04030201u, 0x08070605u, 0x0C0B0A09u,
                       0xBF810000u}));
    // Negative controls: the exact MIMG BVH word appearing as a 32-bit literal constant must NOT trigger!
    EXPECT(!scan_words({0xBE8003FFu, 0xF1980001u, 0x00010000u, 0xBF810000u}));  // SOP1 s_mov_b32 s0, lit
    EXPECT(!scan_words({0x7E0002FFu, 0xF19C0001u, 0x00010000u, 0xBF810000u}));  // VOP1 v_mov_b32 v0, lit
    EXPECT(!scan_words({(0x2Du << 25) | (1u << 9) | 258u, 0xF1980001u, 0x00010000u, 0xBF810000u}));  // VOP2 v_fmaak_f32

    // Full FSSM container slicing (code at u64[0xC0] + 0x28, trailing constant-data block excluded).
    leonida::ComputeProgram prog;
    prog.kind = leonida::ComputeProgram::Kind::fssm;
    prog.bytes = make_synthetic_fssm({0xF1980001u, 0x00010000u, 0xBF810000u});
    EXPECT(rt::d3d12::compute_program_walks_bvh_for_test(prog));
    // Trailing constant-data block after real s_endpgm: s_endpgm, N=2, 2 data words (matching MIMG BVH), closing s_endpgm.
    prog.bytes = make_synthetic_fssm({0xBE800381u, 0xBF810000u, 2u, 0xF1980001u, 0x00010000u, 0xBF810000u});
    EXPECT(!rt::d3d12::compute_program_walks_bvh_for_test(prog));
    // Tiny dummy FSSM (< 0xC8 bytes, like synthetic test blobs): never misclassified.
    prog.bytes = {'F', 'S', 'S', 'M', 0, 0, 0, 0, 0x01, 0x00, 0x98, 0xF1, 0, 0, 0x01, 0};
    EXPECT(!rt::d3d12::compute_program_walks_bvh_for_test(prog));

    // LEONIDA_BVH_HOLD policy parser & Wave32/Wave64 matching.
    leonida::SubgroupSizeCaps wave32{}, wave64{}, unknown{};
    wave32.min_size = 32;
    wave32.max_size = 32;
    wave32.default_size = 32;
    wave64.min_size = 32;
    wave64.max_size = 64;
    wave64.default_size = 64;

    const std::string_view name = "CS_IrradianceCache_ShadowTrace_WithFar";
    const std::string_view sha = "a1b2c3d4e5f60123456789abcdef0123456789abcdef0123456789abcdef0123";
    const char* reason = nullptr;

    // Unset: defaults to auto on Wave32, off on Wave64 and unknown subgroup caps.
    EXPECT(rt::d3d12::evaluate_bvh_hold_policy(nullptr, name, sha, true, wave32, true, &reason) && reason);
    EXPECT(!rt::d3d12::evaluate_bvh_hold_policy(nullptr, name, sha, false, wave32, true));
    EXPECT(!rt::d3d12::evaluate_bvh_hold_policy(nullptr, name, sha, true, wave32, false));
    EXPECT(!rt::d3d12::evaluate_bvh_hold_policy(nullptr, name, sha, true, wave64, true));
    EXPECT(!rt::d3d12::evaluate_bvh_hold_policy(nullptr, name, sha, true, unknown, true));

    // 0 / off / none: hold nothing even on Wave32 with unimplemented rehydration.
    for (const char* v : {"0", "off", "OFF", "none", " false "})
        EXPECT(!rt::d3d12::evaluate_bvh_hold_policy(v, name, sha, true, wave32, true));

    // 1 / all / *: hold every BVH-walking compute shader regardless of Wave64 or rehydration status.
    for (const char* v : {"1", "all", "ALL", "*"}) {
        EXPECT(rt::d3d12::evaluate_bvh_hold_policy(v, name, sha, true, wave64, false));
        EXPECT(!rt::d3d12::evaluate_bvh_hold_policy(v, name, sha, false, wave32, true));
    }

    // auto: hold BVH-walking compute shaders iff rehydrate is unimplemented.
    EXPECT(rt::d3d12::evaluate_bvh_hold_policy("auto", name, sha, true, wave32, true));
    EXPECT(!rt::d3d12::evaluate_bvh_hold_policy("auto", name, sha, true, wave32, false));
    EXPECT(rt::d3d12::evaluate_bvh_hold_policy("AUTO", name, sha, true, wave64, true));
    EXPECT(!rt::d3d12::evaluate_bvh_hold_policy("auto", name, sha, true, wave64, false));

    // Comma-separated name substrings and >=4-char hex sha256 prefixes (case-insensitive hex).
    EXPECT(rt::d3d12::evaluate_bvh_hold_policy("ShadowTrace, deadbeef", name, sha, false, wave64, false));
    EXPECT(rt::d3d12::evaluate_bvh_hold_policy("OtherShader, A1B2C3D4", "CS_Other", sha, false, wave64, false));
    EXPECT(!rt::d3d12::evaluate_bvh_hold_policy("OtherShader, deadbeef", name, sha, true, wave32, true));

    std::puts("bvh hold: RDNA2 MIMG BVH scanner (0xE6/0xE7 + literal guards) and LEONIDA_BVH_HOLD policies ok");
    return 0;
}

int submit_empty(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64& fence_value) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(SUCCEEDED(queue->Signal(fence, ++fence_value)));
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
    return 0;
}

// Blockers 2A + 2B: API buffer views (CBV, placed SRV, placed UAV) targeting initially-unbacked title memory inside a
// console descriptor heap preserve their guest VA in title memory while refresh_console_heap zeros heap.copy, and
// advancing leonida::title_memory_map::generation() sweeps empty_slots immediately instead of waiting 32 submissions.
int check_console_api_and_generation_recovery(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* fence,
                                              UINT64& fence_value, std::uint64_t unmapped,
                                              std::vector<void*>& deferred_frees) {
    constexpr UINT slots = 256;
    constexpr std::uint64_t heap_bytes = std::uint64_t(slots) * kDescriptor;
    auto* title = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, heap_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    EXPECT(title);
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots,
                                          D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(rt::d3d12::create_console_descriptor_heap(device, &desc, reinterpret_cast<std::uint64_t>(title),
                                                               heap_bytes, IID_PPV_ARGS(&heap.p))));
    D3D12_DESCRIPTOR_HEAP_DESC got{};
    D3D12_GPU_VIRTUAL_ADDRESS shader_va = 0;
    EXPECT(rt::d3d12::descriptor_heap_info(heap.p, got, shader_va));

    // Reserved (uncommitted) title memory: initially unbacked.
    void* pages = VirtualAlloc(nullptr, 65536, MEM_TOP_DOWN | MEM_RESERVE, PAGE_READWRITE);
    EXPECT(pages);
    deferred_frees.push_back(pages);
    const auto target_va = reinterpret_cast<std::uint64_t>(pages);
    auto& aliases = leonida::title_va_aliases();
    aliases.add_title_memory(target_va, 65536);

    const auto handle = [&](UINT slot) {
        return D3D12_CPU_DESCRIPTOR_HANDLE{reinterpret_cast<SIZE_T>(title + std::size_t(slot) * kDescriptor)};
    };
    const auto title_words = [&](UINT slot) {
        return reinterpret_cast<const std::uint32_t*>(title + std::size_t(slot) * kDescriptor);
    };

    // Slot 0: CBV targeting uncommitted title memory -> guest VA preserved in title heap.
    const D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{target_va, kViewBytes};
    device->CreateConstantBufferView(&cbv, handle(0));
    EXPECT(base_of(title_words(0)) == (target_va & 0xFFFFFFFFFFFFull) && title_words(0)[2] != 0);

    // Slot 1: placed raw buffer SRV targeting uncommitted title memory -> guest VA preserved in title heap.
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = kViewBytes / 4;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    EXPECT(rt::d3d12::write_placed_buffer_view(device, false, &srv, target_va + 0x1000, kViewBytes, handle(1)));
    EXPECT(base_of(title_words(1)) == ((target_va + 0x1000) & 0xFFFFFFFFFFFFull) && title_words(1)[2] == kViewBytes);

    // Slot 2: placed raw buffer UAV targeting uncommitted title memory -> guest VA preserved in title heap.
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = kViewBytes / 4;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    EXPECT(rt::d3d12::write_placed_buffer_view(device, true, &uav, target_va + 0x2000, kViewBytes, handle(2)));
    EXPECT(base_of(title_words(2)) == ((target_va + 0x2000) & 0xFFFFFFFFFFFFull) && title_words(2)[2] == kViewBytes);

    // Slot 3 (negative control): non-title unmapped address in console heap MUST still be zeroed immediately.
    const D3D12_CONSTANT_BUFFER_VIEW_DESC non_title{unmapped, kViewBytes};
    device->CreateConstantBufferView(&non_title, handle(3));
    EXPECT(zero_size(title_words(3)));

    // Slots 4..67 (64 slots): CPU-written raw V#s targeting the same uncommitted title memory.
    const auto cold_vsharp = raw_vsharp(target_va + 0x3000);
    for (UINT slot = 4; slot < 68; ++slot)
        std::memcpy(title + std::size_t(slot) * kDescriptor, cold_vsharp.data(), sizeof(cold_vsharp));

    // First submission: refresh_console_heap must zero every unbacked slot in heap.copy while preserving title bytes.
    if (submit_empty(device, queue, fence, fence_value)) return 1;
    for (UINT slot = 0; slot < 68; ++slot) {
        std::uint32_t gpu_w[8]{};
        EXPECT(rt::d3d12::read_arena(device, shader_va + std::uint64_t(slot) * kDescriptor, gpu_w, sizeof(gpu_w)));
        EXPECT(zero_size(gpu_w));
    }
    EXPECT(base_of(title_words(0)) == (target_va & 0xFFFFFFFFFFFFull));
    EXPECT(base_of(title_words(1)) == ((target_va + 0x1000) & 0xFFFFFFFFFFFFull));
    EXPECT(base_of(title_words(2)) == ((target_va + 0x2000) & 0xFFFFFFFFFFFFull));

    // Commit the title pages and bump title_memory_map::generation(), without touching title descriptor bytes.
    EXPECT(VirtualAlloc(pages, 65536, MEM_COMMIT, PAGE_READWRITE) == pages);
    leonida::title_memory_map::changed(pages, 65536);

    // Generation-triggered sweep recovers slots immediately (well before the 32-submit fallback).
    unsigned recovered = 0;
    for (unsigned pass = 0; pass < 4; ++pass) {
        if (submit_empty(device, queue, fence, fence_value)) return 1;
        recovered = 0;
        for (UINT slot = 0; slot < 68; ++slot) {
            if (slot == 3) continue;
            std::uint32_t gpu_w[8]{};
            EXPECT(rt::d3d12::read_arena(device, shader_va + std::uint64_t(slot) * kDescriptor, gpu_w, sizeof(gpu_w)));
            if (base_of(gpu_w) != 0 && gpu_w[2] != 0) ++recovered;
        }
        if (pass == 0) EXPECT(recovered >= 16);
        if (recovered == 67) break;
    }
    EXPECT(recovered == 67);
    for (UINT slot : {0u, 1u, 2u}) {
        std::uint32_t gpu_w[8]{};
        EXPECT(rt::d3d12::read_arena(device, shader_va + std::uint64_t(slot) * kDescriptor, gpu_w, sizeof(gpu_w)));
        const std::uint64_t guest = target_va + std::uint64_t(slot) * 0x1000;
        EXPECT(base_of(gpu_w) == (aliases.shader_va(guest) & 0xFFFFFFFFFFFFull) && gpu_w[2] != 0);
    }

    std::printf("console API + generation recovery: CBV/SRV/UAV + 64 cold slots recovered (%u/67) after map change\n",
                recovered);
    heap.reset();
    aliases.remove_title_memory(target_va);
    VirtualFree(title, 0, MEM_RELEASE);
    return 0;
}

// Blocker 2C: with LEONIDA_GEOMETRY_MB > 0 (VsharpPath::scratch), valid MEM_PRIVATE V#s <= 16 MiB must still obtain
// title windows instead of being silently zeroed, while non-canonical untyped garbage and >16 MiB spans stay zeroed.
int run_scratch_mode() {
    _putenv_s("LEONIDA_GEOMETRY_MB", "16");
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);

    std::vector<void*> deferred_frees;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.subgroup = d.subgroup;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        const D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        UINT64 fence_value = 0;

        constexpr UINT slots = 64;
        constexpr std::uint64_t heap_bytes = std::uint64_t(slots) * kDescriptor;
        auto* title = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, heap_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        EXPECT(title);
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots,
                                              D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        EXPECT(SUCCEEDED(rt::d3d12::create_console_descriptor_heap(device.p, &desc, reinterpret_cast<std::uint64_t>(title),
                                                                   heap_bytes, IID_PPV_ARGS(&heap.p))));
        D3D12_DESCRIPTOR_HEAP_DESC got{};
        D3D12_GPU_VIRTUAL_ADDRESS shader_va = 0;
        EXPECT(rt::d3d12::descriptor_heap_info(heap.p, got, shader_va));

        auto& aliases = leonida::title_va_aliases();
        const auto alloc_private = [&](std::size_t bytes, bool commit) {
            void* p = VirtualAlloc(nullptr, bytes, MEM_TOP_DOWN | MEM_RESERVE | (commit ? MEM_COMMIT : 0), PAGE_READWRITE);
            if (p) {
                deferred_frees.push_back(p);
                aliases.add_title_memory(reinterpret_cast<std::uint64_t>(p), bytes);
            }
            return p;
        };

        void* priv_fmt = alloc_private(65536, true);
        void* priv_uav = alloc_private(65536, true);
        void* priv_deferred = alloc_private(65536, false);
        void* priv_garbage = alloc_private(65536, true);
        constexpr std::size_t kOverWindow = 20u << 20;  // > kTitleWindowMax (16 MiB)
        void* priv_big = alloc_private(kOverWindow, true);
        EXPECT(priv_fmt && priv_uav && priv_deferred && priv_garbage && priv_big);

        const auto va_fmt = reinterpret_cast<std::uint64_t>(priv_fmt);
        const auto va_uav = reinterpret_cast<std::uint64_t>(priv_uav);
        const auto va_deferred = reinterpret_cast<std::uint64_t>(priv_deferred);
        const auto va_garbage = reinterpret_cast<std::uint64_t>(priv_garbage);
        const auto va_big = reinterpret_cast<std::uint64_t>(priv_big);

        // Slot 0: formatted V# (format = R32_UINT) pointing to MEM_PRIVATE title memory.
        const auto w_fmt = raw_vsharp(va_fmt);
        std::memcpy(title + 0 * kDescriptor, w_fmt.data(), sizeof(w_fmt));

        // Slot 1: canonical raw UAV V# (format == 0, encode(desc) == words) pointing to MEM_PRIVATE title memory.
        leonida::gfx10::Buffer uav_buf{};
        uav_buf.address = va_uav;
        uav_buf.num_records = kViewBytes;
        uav_buf.format = 0;
        uav_buf.oob_select = leonida::gfx10::OobSelect::raw;
        const auto w_uav = leonida::gfx10::encode(uav_buf);
        std::memcpy(title + 1 * kDescriptor, w_uav.data(), sizeof(w_uav));

        // Slot 2: API CBV targeting initially-uncommitted MEM_PRIVATE title memory (2A + 2B + 2C combined).
        const D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{va_deferred, kViewBytes};
        device->CreateConstantBufferView(&cbv, D3D12_CPU_DESCRIPTOR_HANDLE{reinterpret_cast<SIZE_T>(title + 2 * kDescriptor)});

        // Slot 3 (negative control): untyped (format == 0) non-canonical garbage word (extra reserved bit 23 set).
        auto w_garbage = uav_buf;
        w_garbage.address = va_garbage;
        auto garbage_words = leonida::gfx10::encode(w_garbage);
        garbage_words[3] |= 1u << 23;
        std::memcpy(title + 3 * kDescriptor, garbage_words.data(), sizeof(garbage_words));

        // Slot 4 (negative control): MEM_PRIVATE V# larger than kTitleWindowMax (20 MiB > 16 MiB).
        leonida::gfx10::Buffer big_buf{};
        big_buf.address = va_big;
        big_buf.num_records = std::uint32_t(kOverWindow);
        big_buf.format = leonida::gfx10::format_from_vulkan(VK_FORMAT_R32_UINT);
        big_buf.oob_select = leonida::gfx10::OobSelect::raw;
        const auto w_big = leonida::gfx10::encode(big_buf);
        std::memcpy(title + 4 * kDescriptor, w_big.data(), sizeof(w_big));

        if (submit_empty(device.p, queue.p, fence.p, fence_value)) return 1;
        auto read_slot = [&](UINT slot, std::uint32_t (&out)[8]) {
            return rt::d3d12::read_arena(device.p, shader_va + std::uint64_t(slot) * kDescriptor, out, sizeof(out));
        };
        std::uint32_t s0[8]{}, s1[8]{}, s2[8]{}, s3[8]{}, s4[8]{};
        EXPECT(read_slot(0, s0) && read_slot(1, s1) && read_slot(2, s2) && read_slot(3, s3) && read_slot(4, s4));
        EXPECT(base_of(s0) != 0 && base_of(s0) != (va_fmt & 0xFFFFFFFFFFFFull) && s0[2] == kViewBytes);
        EXPECT(base_of(s1) != 0 && base_of(s1) != (va_uav & 0xFFFFFFFFFFFFull) && s1[2] == kViewBytes);
        EXPECT(zero_size(s2));  // uncommitted on submit 1
        EXPECT(zero_size(s3));  // non-canonical format==0 garbage rejected
        EXPECT(zero_size(s4));  // > 16 MiB MEM_PRIVATE rejected

        // Commit slot 2's MEM_PRIVATE pages and advance generation -> recovered via VsharpPath::scratch window.
        EXPECT(VirtualAlloc(priv_deferred, 65536, MEM_COMMIT, PAGE_READWRITE) == priv_deferred);
        leonida::title_memory_map::changed(priv_deferred, 65536);
        if (submit_empty(device.p, queue.p, fence.p, fence_value)) return 1;
        EXPECT(read_slot(2, s2));
        EXPECT(base_of(s2) != 0 && base_of(s2) != (va_deferred & 0xFFFFFFFFFFFFull) && s2[2] != 0);

        std::puts("scratch V# path (LEONIDA_GEOMETRY_MB=16): MEM_PRIVATE V#s window-backed; garbage and >16 MiB zeroed");
        heap.reset();
        for (void* p : deferred_frees) aliases.remove_title_memory(reinterpret_cast<std::uint64_t>(p));
        VirtualFree(title, 0, MEM_RELEASE);
    }
    for (void* p : deferred_frees) VirtualFree(p, 0, MEM_RELEASE);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    return 0;
}

int run_scratch_subprocess() {
    wchar_t exe[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    EXPECT(n > 0 && n < MAX_PATH);
    std::wstring cmd = L"\"";
    cmd.append(exe, n);
    cmd.append(L"\" --scratch");
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    EXPECT(CreateProcessW(exe, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi));
    EXPECT(WaitForSingleObject(pi.hProcess, 60000) == WAIT_OBJECT_0);
    DWORD exit_code = 1;
    EXPECT(GetExitCodeProcess(pi.hProcess, &exit_code));
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    EXPECT(exit_code == 0 || exit_code == 77);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--scratch") == 0) return run_scratch_mode();
    if (check_bvh_hold_and_scanner()) return 1;
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);
    std::vector<void*> deferred_frees;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.subgroup = d.subgroup;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        const std::uint64_t arena = rt::d3d12::device_arena_size(device.p);
        EXPECT(arena > 0);
        const std::uint64_t unmapped = unmapped_address();
        EXPECT(unmapped > arena && unmapped < (1ull << 47));
        std::uint64_t lo = 0, hi = 0;
        EXPECT(!leonida::title_va_aliases().title_memory(unmapped, lo, hi));

        const D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        UINT64 fence_value = 0;
        Buffers b;
        if (make_buffers(device.p, arena, b)) return 1;
        auto& aliases = leonida::title_va_aliases();
        EXPECT(aliases.resolve(unmapped, kViewBytes, leonida::TitleVaAliases::root_view) == 0);
        EXPECT(aliases.resolve(b.upload_va, kViewBytes, leonida::TitleVaAliases::root_view) == b.upload_va);
        if (check_created_views(device.p, b, unmapped)) return 1;
        if (check_inline_table(device.p, queue.p, fence.p, fence_value, b, unmapped)) return 1;
        if (check_console_heap(device.p, queue.p, fence.p, fence_value, b, unmapped)) return 1;
        if (check_layer_heap_guard(device.p)) return 1;
        if (check_console_api_and_generation_recovery(device.p, queue.p, fence.p, fence_value, unmapped,
                                                      deferred_frees))
            return 1;
    }
    for (void* p : deferred_frees) VirtualFree(p, 0, MEM_RELEASE);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    if (run_scratch_subprocess()) return 1;
    std::puts("ok");
    return 0;
}
