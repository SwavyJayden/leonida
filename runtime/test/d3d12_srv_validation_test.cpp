// Validation of CreateShaderResourceView descriptors across texture and buffer dimensions:
//   - Shader4ComponentMapping (always-set bit 12 and each selector <= FORCE_VALUE_1) for all texture and buffer SRVs;
//   - ResourceMinLODClamp (non-negative and non-NaN) across 1D, 2D, 2D array, 3D, cube and cube-array SRVs;
//   - Typeless resources and view formats (null desc, DXGI_FORMAT_UNKNOWN, or a typeless view format leave null);
//   - MipLevels, ArraySize and NumCubes bounds without silent clamping or 32-bit NumCubes * 6 overflow;
//   - Positive controls checking GFX10 T#/V# decode and bindless hash registration, and 0 Vulkan validation errors.
// No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#define EXPECT(condition)                                                               \
    do {                                                                                \
        if (!(condition)) {                                                             \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);                \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);

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

void* vtable_slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

D3D12_RESOURCE_DESC make_buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC make_tex_desc(D3D12_RESOURCE_DIMENSION dim, UINT64 width, UINT height, UINT16 depth_or_array,
                                  UINT16 mips, DXGI_FORMAT format,
                                  D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = dim;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = depth_or_array;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no Vulkan device\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;

    UINT valid_count = 0;
    UINT rejected_count = 0;

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.texture_compression_bc = d.bc;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.sampler_filter_minmax = d.sampler_filter_minmax;
        info.custom_border_color = d.custom_border_color;
        info.custom_border_color_without_format = d.custom_border_color_without_format;

        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(step >= sizeof(leonida::gfx10::ImageWords));
        Ref<ID3D12DescriptorHeap> heap;
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        const std::uint64_t heap_cpu = reinterpret_cast<XboxHeapHandleFn>(vtable_slot(heap.p, 9))(heap.p);
        EXPECT(heap_cpu != 0);

        const auto slot_handle = [&](UINT idx) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(heap_cpu + idx * step)}; };
        const auto slot_bytes = [&](UINT idx) { return reinterpret_cast<std::uint8_t*>(slot_handle(idx).ptr); };

        const auto expect_valid_tex = [&](ID3D12Resource* res, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
                                          leonida::gfx10::Image& out) -> bool {
            std::memset(slot_bytes(0), 0xAA, step);
            device->CreateShaderResourceView(res, desc, slot_handle(0));
            leonida::gfx10::ImageWords words{};
            std::memcpy(words.data(), slot_bytes(0), sizeof(words));
            if (!leonida::gfx10::is_image(words) || !leonida::gfx10::decode(words, out)) return false;
            if (d.bindless) {
                std::uint32_t key[8]{};
                std::memcpy(key, words.data(), sizeof(key));
                if (rt::d3d12::bindless_hash_lookup(device.p, key) == 0) return false;
            }
            ++valid_count;
            return true;
        };

        const auto expect_valid_buf = [&](ID3D12Resource* res, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
                                          leonida::gfx10::Buffer& out) -> bool {
            std::memset(slot_bytes(0), 0xAA, step);
            device->CreateShaderResourceView(res, desc, slot_handle(0));
            leonida::gfx10::BufferWords words{};
            std::memcpy(words.data(), slot_bytes(0), sizeof(words));
            if (!leonida::gfx10::decode(words, out)) return false;
            if (!std::all_of(slot_bytes(0) + sizeof(words), slot_bytes(0) + step,
                             [](std::uint8_t b) { return b == 0; }))
                return false;
            ++valid_count;
            return true;
        };

        const auto expect_null_srv = [&](ID3D12Resource* res, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc) -> bool {
            std::memset(slot_bytes(0), 0xAA, step);
            device->CreateShaderResourceView(res, desc, slot_handle(0));
            if (!std::all_of(slot_bytes(0), slot_bytes(0) + step, [](std::uint8_t b) { return b == 0; })) return false;
            ++rejected_count;
            return true;
        };

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        Ref<ID3D12Resource> tex1d, tex2d, tex2d_arr18, tex2d_arr4, tex3d;
        Ref<ID3D12Resource> typeless_2d, typeless_arr2, buf;

        const auto d_1d = make_tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 16, 1, 1, 3, DXGI_FORMAT_R8G8B8A8_UNORM);
        const auto d_2d = make_tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 3, DXGI_FORMAT_R8G8B8A8_UNORM);
        const auto d_arr18 = make_tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 8, 8, 18, 3, DXGI_FORMAT_R8G8B8A8_UNORM);
        const auto d_arr4 = make_tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 8, 8, 4, 1, DXGI_FORMAT_R8G8B8A8_UNORM);
        const auto d_3d = make_tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 8, 8, 4, 2, DXGI_FORMAT_R8G8B8A8_UNORM);
        const auto d_t2d = make_tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 8, 8, 1, 2, DXGI_FORMAT_R8G8B8A8_TYPELESS,
                                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        const auto d_tarr = make_tex_desc(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 8, 8, 2, 2, DXGI_FORMAT_R8G8B8A8_TYPELESS);
        const auto d_buf = make_buffer_desc(1024);

        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_1d,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex1d.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_2d,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex2d.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_arr18,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&tex2d_arr18.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_arr4,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&tex2d_arr4.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_3d,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex3d.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_t2d,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&typeless_2d.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_tarr,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&typeless_arr2.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d_buf,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf.p))));

        // --- Null resource always leaves null descriptor.
        D3D12_SHADER_RESOURCE_VIEW_DESC base2d{};
        base2d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        base2d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        base2d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        base2d.Texture2D = {0, UINT_MAX, 0, 0.0f};
        EXPECT(expect_null_srv(nullptr, nullptr));
        EXPECT(expect_null_srv(nullptr, &base2d));

        // --- Positive controls across dimensions.
        leonida::gfx10::Image img{};
        EXPECT(expect_valid_tex(tex1d.p, nullptr, img));
        EXPECT(img.first_mip == 0 && img.mip_count == 3);

        D3D12_SHADER_RESOURCE_VIEW_DESC srv1d{};
        srv1d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv1d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
        srv1d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv1d.Texture1D = {1, UINT_MAX, -0.0f};
        EXPECT(expect_valid_tex(tex1d.p, &srv1d, img));
        EXPECT(img.first_mip == 1 && img.mip_count == 2);
        srv1d.Texture1D = {0, 2, 5.0f};
        EXPECT(expect_valid_tex(tex1d.p, &srv1d, img));
        EXPECT(img.mip_count == 2);

        EXPECT(expect_valid_tex(tex2d.p, nullptr, img));
        EXPECT(expect_valid_tex(tex2d.p, &base2d, img));
        EXPECT(img.first_mip == 0 && img.mip_count == 3);
        EXPECT(expect_valid_tex(typeless_2d.p, &base2d, img));
        EXPECT(img.mip_count == 2);
        if (d.image_view_min_lod) {
            D3D12_SHADER_RESOURCE_VIEW_DESC minlod = base2d;
            minlod.Texture2D.ResourceMinLODClamp = 0.75f;
            EXPECT(expect_valid_tex(tex2d.p, &minlod, img));
            EXPECT(img.min_lod == 0.75f);
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC srv2d_arr{};
        srv2d_arr.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv2d_arr.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv2d_arr.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv2d_arr.Texture2DArray = {0, 3, 2, 4, 0, 0.0f};
        EXPECT(expect_valid_tex(tex2d_arr18.p, &srv2d_arr, img));
        EXPECT(img.first_layer == 2 && img.layer_count == 4);
        srv2d_arr.Texture2DArray = {1, UINT_MAX, 16, UINT_MAX, 0, 0.0f};
        EXPECT(expect_valid_tex(tex2d_arr18.p, &srv2d_arr, img));
        EXPECT(img.first_mip == 1 && img.mip_count == 2 && img.first_layer == 16 && img.layer_count == 2);

        D3D12_SHADER_RESOURCE_VIEW_DESC srv3d{};
        srv3d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv3d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        srv3d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv3d.Texture3D = {1, 1, 0.0f};
        EXPECT(expect_valid_tex(tex3d.p, &srv3d, img));
        EXPECT(img.first_mip == 1 && img.mip_count == 1);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvcube{};
        srvcube.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srvcube.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        srvcube.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvcube.TextureCube = {1, 2, 0.0f};
        EXPECT(expect_valid_tex(tex2d_arr18.p, &srvcube, img));
        EXPECT(img.first_mip == 1 && img.mip_count == 2 && img.layer_count == 6);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvcubearr{};
        srvcubearr.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srvcubearr.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        srvcubearr.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvcubearr.TextureCubeArray = {0, 3, 6, 2, 0.0f};
        EXPECT(expect_valid_tex(tex2d_arr18.p, &srvcubearr, img));
        EXPECT(img.first_layer == 6 && img.layer_count == 12);
        srvcubearr.TextureCubeArray = {0, 3, 6, UINT_MAX, 0.0f};
        EXPECT(expect_valid_tex(tex2d_arr18.p, &srvcubearr, img));
        EXPECT(img.first_layer == 6 && img.layer_count == 12);

        leonida::gfx10::Buffer bview{};
        D3D12_SHADER_RESOURCE_VIEW_DESC srvbuf_raw{};
        srvbuf_raw.Format = DXGI_FORMAT_R32_TYPELESS;
        srvbuf_raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvbuf_raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvbuf_raw.Buffer = {0, 32, 0, D3D12_BUFFER_SRV_FLAG_RAW};
        EXPECT(expect_valid_buf(buf.p, &srvbuf_raw, bview));

        D3D12_SHADER_RESOURCE_VIEW_DESC srvbuf_struct{};
        srvbuf_struct.Format = DXGI_FORMAT_UNKNOWN;
        srvbuf_struct.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvbuf_struct.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvbuf_struct.Buffer = {0, 16, 16, D3D12_BUFFER_SRV_FLAG_NONE};
        EXPECT(expect_valid_buf(buf.p, &srvbuf_struct, bview));

        D3D12_SHADER_RESOURCE_VIEW_DESC srvbuf_typed{};
        srvbuf_typed.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srvbuf_typed.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvbuf_typed.Shader4ComponentMapping =
            D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(2, 1, 0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1);
        srvbuf_typed.Buffer = {0, 16, 0, D3D12_BUFFER_SRV_FLAG_NONE};
        EXPECT(expect_valid_buf(buf.p, &srvbuf_typed, bview));

        // --- Invalid Shader4ComponentMapping across all dimensions (textures + buffers).
        const UINT bad_mappings[] = {
            0x0688u,  // missing D3D12 always-set bit 12
            0x3688u,  // unexpected bit 13 set
            D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(6, 1, 2, 3),
            D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 1, 7, 3),
        };
        for (UINT mapping : bad_mappings) {
            D3D12_SHADER_RESOURCE_VIEW_DESC bad = srv1d;
            bad.Texture1D = {0, 1, 0.0f};
            bad.Shader4ComponentMapping = mapping;
            EXPECT(expect_null_srv(tex1d.p, &bad));

            bad = base2d;
            bad.Shader4ComponentMapping = mapping;
            EXPECT(expect_null_srv(tex2d.p, &bad));

            bad = srv2d_arr;
            bad.Texture2DArray = {0, 1, 0, 1, 0, 0.0f};
            bad.Shader4ComponentMapping = mapping;
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));

            bad = srv3d;
            bad.Shader4ComponentMapping = mapping;
            EXPECT(expect_null_srv(tex3d.p, &bad));

            bad = srvcube;
            bad.Shader4ComponentMapping = mapping;
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));

            bad = srvcubearr;
            bad.TextureCubeArray = {0, 1, 0, 1, 0.0f};
            bad.Shader4ComponentMapping = mapping;
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));

            bad = srvbuf_raw;
            bad.Shader4ComponentMapping = mapping;
            EXPECT(expect_null_srv(buf.p, &bad));

            bad = srvbuf_struct;
            bad.Shader4ComponentMapping = mapping;
            EXPECT(expect_null_srv(buf.p, &bad));

            bad = srvbuf_typed;
            bad.Shader4ComponentMapping = mapping;
            EXPECT(expect_null_srv(buf.p, &bad));
        }

        // --- Invalid ResourceMinLODClamp (negative or NaN).
        const float bad_lods[] = {
            -0.25f,
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
        };
        for (float bad_lod : bad_lods) {
            D3D12_SHADER_RESOURCE_VIEW_DESC bad = srv1d;
            bad.Texture1D = {0, 2, bad_lod};
            EXPECT(expect_null_srv(tex1d.p, &bad));

            bad = base2d;
            bad.Texture2D.ResourceMinLODClamp = bad_lod;
            EXPECT(expect_null_srv(tex2d.p, &bad));

            bad = srv2d_arr;
            bad.Texture2DArray = {0, 2, 0, 1, 0, bad_lod};
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));

            bad = srv3d;
            bad.Texture3D.ResourceMinLODClamp = bad_lod;
            EXPECT(expect_null_srv(tex3d.p, &bad));

            bad = srvcube;
            bad.TextureCube.ResourceMinLODClamp = bad_lod;
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));

            bad = srvcubearr;
            bad.TextureCubeArray = {0, 2, 0, 1, bad_lod};
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));
        }

        // --- Typeless resource / view format rules.
        EXPECT(expect_null_srv(typeless_2d.p, nullptr));
        EXPECT(expect_null_srv(typeless_arr2.p, nullptr));
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC bad = base2d;
            for (DXGI_FORMAT fmt : {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_TYPELESS}) {
                bad.Format = fmt;
                EXPECT(expect_null_srv(typeless_2d.p, &bad));
            }
            bad.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
            EXPECT(expect_null_srv(tex2d.p, &bad));  // typeless view format on typed resource

            bad = srv2d_arr;
            bad.Texture2DArray = {0, 1, 0, 1, 0, 0.0f};
            for (DXGI_FORMAT fmt : {DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R8G8B8A8_TYPELESS}) {
                bad.Format = fmt;
                EXPECT(expect_null_srv(typeless_arr2.p, &bad));
            }
        }
        // Typeless resource UAV with null desc must also remain null.
        std::memset(slot_bytes(0), 0xAA, step);
        device->CreateUnorderedAccessView(typeless_2d.p, nullptr, nullptr, slot_handle(0));
        EXPECT(std::all_of(slot_bytes(0), slot_bytes(0) + step, [](std::uint8_t b) { return b == 0; }));
        ++rejected_count;

        // --- Out-of-bounds MipLevels (0, overflows mip range, or UINT_MAX - 1).
        for (const auto& [first_mip, mip_count] :
             {std::pair{0u, 0u}, std::pair{1u, 3u}, std::pair{1u, UINT_MAX - 1u}}) {
            D3D12_SHADER_RESOURCE_VIEW_DESC bad = srv1d;
            bad.Texture1D = {first_mip, mip_count, 0.0f};
            EXPECT(expect_null_srv(tex1d.p, &bad));

            bad = base2d;
            bad.Texture2D = {first_mip, mip_count, 0, 0.0f};
            EXPECT(expect_null_srv(tex2d.p, &bad));

            bad = srv2d_arr;
            bad.Texture2DArray = {first_mip, mip_count, 0, 1, 0, 0.0f};
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));

            bad = srv3d;
            bad.Texture3D = {first_mip, mip_count, 0.0f};
            EXPECT(expect_null_srv(tex3d.p, &bad));

            bad = srvcube;
            bad.TextureCube = {first_mip, mip_count, 0.0f};
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));

            bad = srvcubearr;
            bad.TextureCubeArray = {first_mip, mip_count, 0, 1, 0.0f};
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));
        }

        // --- Out-of-bounds ArraySize on 2D array.
        for (const auto& [first_slice, slice_count] :
             {std::pair{0u, 0u}, std::pair{16u, 3u}, std::pair{1u, UINT_MAX - 1u}}) {
            D3D12_SHADER_RESOURCE_VIEW_DESC bad = srv2d_arr;
            bad.Texture2DArray = {0, 1, first_slice, slice_count, 0, 0.0f};
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));
        }

        // --- TEXTURECUBE on a 4-slice (< 6) 2D array must fail cleanly.
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC bad = srvcube;
            bad.TextureCube = {0, 1, 0.0f};
            EXPECT(expect_null_srv(tex2d_arr4.p, &bad));
        }

        // --- TEXTURECUBEARRAY bounds and NumCubes * 6 overflow regression check.
        for (const auto& [first_face, num_cubes] :
             {std::pair{0u, 0u}, std::pair{12u, 2u}, std::pair{14u, UINT_MAX}, std::pair{0u, 0x30000000u}}) {
            D3D12_SHADER_RESOURCE_VIEW_DESC bad = srvcubearr;
            bad.TextureCubeArray = {0, 1, first_face, num_cubes, 0.0f};
            EXPECT(expect_null_srv(tex2d_arr18.p, &bad));
        }
    }

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", d.validation_errors());
        return 1;
    }

    std::printf("d3d12_srv_validation: %u valid cases accepted, %u invalid cases rejected (0 validation errors)\n",
                valid_count, rejected_count);
    return 0;
}
