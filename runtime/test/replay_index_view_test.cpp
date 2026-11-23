// An indexed draw's index buffer view SizeInBytes bounds the index fetch (D3D12: a fetch past the view reads 0).
// prepare_draw decides, from the view and the draw's StartIndexLocation/IndexCount, whether to bind the buffer, the
// zero page, or a staged copy that holds only the in-view indices. No Vulkan loader, GPU or game data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>

namespace {

void dummy() {}

template<typename T>
T stub() {
    return reinterpret_cast<T>(&dummy);
}

template<typename T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

constexpr std::uint64_t ib_va = 0x10000;
constexpr std::uint32_t dxgi_r16_uint = 57;

}  // namespace

int main() {
    using namespace leonida;
    using detail::PreparedDraw;

    const VkBuffer index_buffer = handle<VkBuffer>(0x1111);
    const VkBuffer zero_buffer = handle<VkBuffer>(0x2222);
    const VkBuffer scratch_buffer = handle<VkBuffer>(0x3333);

    GraphicsPipeline pipeline;
    pipeline.pipeline = handle<VkPipeline>(0x4444);

    ReplayFunctions vk{};
    vk.begin_rendering = stub<PFN_vkCmdBeginRendering>();
    vk.end_rendering = stub<PFN_vkCmdEndRendering>();
    vk.copy_buffer = stub<PFN_vkCmdCopyBuffer>();
    vk.fill_buffer = stub<PFN_vkCmdFillBuffer>();
    vk.bind_pipeline = stub<PFN_vkCmdBindPipeline>();
    vk.push_constants = stub<PFN_vkCmdPushConstants>();
    vk.bind_vertex_buffers2 = stub<PFN_vkCmdBindVertexBuffers2>();
    vk.bind_index_buffer = stub<PFN_vkCmdBindIndexBuffer>();
    vk.set_viewport = stub<PFN_vkCmdSetViewport>();
    vk.set_scissor = stub<PFN_vkCmdSetScissor>();
    vk.set_primitive_topology = stub<PFN_vkCmdSetPrimitiveTopology>();
    vk.set_primitive_restart_enable = stub<PFN_vkCmdSetPrimitiveRestartEnable>();
    vk.set_stencil_reference = stub<PFN_vkCmdSetStencilReference>();
    vk.set_blend_constants = stub<PFN_vkCmdSetBlendConstants>();
    vk.draw = stub<PFN_vkCmdDraw>();
    vk.draw_indexed = stub<PFN_vkCmdDrawIndexed>();
    vk.graphics_pipeline = [&](std::uintptr_t) { return &pipeline; };
    vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) { return VkImageView(VK_NULL_HANDLE); };
    // The index buffer is 8 indices (16 bytes) of the arena; a view is cut at the buffer's end.
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
        if (va != ib_va) return false;
        size = std::min<std::uint64_t>(size, 16);
        buffer = index_buffer;
        offset = 0;
        return true;
    };
    vk.zero_buffer = zero_buffer;
    vk.zero_offset = 0;
    vk.zero_size = 64;
    vk.copy_scratch = [&](VkDeviceSize, VkBuffer& buffer, VkDeviceSize& offset) {
        buffer = scratch_buffer;
        offset = 0;
        return true;
    };
    const auto no_resource = [](auto) -> const ResourceBinding* { return nullptr; };

    const auto prepare = [&](std::uint64_t va, std::uint32_t view_bytes, std::uint32_t first, std::uint32_t count,
                             PreparedDraw& p) {
        Draw draw;
        draw.indexed = true;
        draw.index = {va, view_bytes, dxgi_r16_uint};
        draw.first = first;
        draw.count = count;
        draw.instances = 1;
        return detail::prepare_draw(draw, no_resource, vk, p);
    };

    {  // View of 4 indices, draw of indices 0..3: wholly inside, the buffer is bound as it is.
        PreparedDraw p;
        expect(prepare(ib_va, 8, 0, 4, p) == ReplayResult::success, "inside: success");
        expect(p.index_buffer == index_buffer && p.index_first == 0 && p.index_stage.fill_bytes == 0,
               "inside: the buffer is bound, no staging");
    }
    {  // View of 4 indices, draw of indices 4..7: all outside, every index reads 0, from the zero page.
        PreparedDraw p;
        expect(prepare(ib_va, 8, 4, 4, p) == ReplayResult::success, "outside: success");
        expect(p.index_buffer == zero_buffer && p.index_first == 0 && p.index_stage.fill_bytes == 0,
               "outside: the zero page is bound, not the buffer's second half");
    }
    {  // View of 4 indices, draw of indices 2..5: 2 and 3 are read, 4 and 5 read 0.
        PreparedDraw p;
        expect(prepare(ib_va, 8, 2, 4, p) == ReplayResult::success, "straddle: success");
        expect(p.index_buffer == scratch_buffer && p.index_first == 0, "straddle: a staged copy is bound");
        expect(p.index_stage.source == index_buffer && p.index_stage.source_offset == 4 &&
                   p.index_stage.copy_bytes == 4,
               "straddle: only the 2 in-view indices (4 bytes at byte 4) are copied");
    }
    {  // No index buffer bound: every index reads 0.
        PreparedDraw p;
        expect(prepare(0, 0, 0, 16, p) == ReplayResult::success, "null view: success");
        expect(p.index_buffer == zero_buffer && p.index_first == 0 && p.index_stage.fill_bytes == 0,
               "null view: the zero page is bound");
    }
    {  // No index buffer bound, a draw larger than the zero page: staged zeros, never the arena after the page.
        PreparedDraw p;
        expect(prepare(0, 0, 0, 100, p) == ReplayResult::success, "null view, large draw: success");
        expect(p.index_buffer == scratch_buffer && p.index_stage.copy_bytes == 0 && p.index_stage.fill_bytes >= 200,
               "null view, large draw: a zeroed scratch covers the whole draw");
    }

    if (failures) {
        std::fprintf(stderr, "replay_index_view_test: FAILED (%d failures)\n", failures);
        return 1;
    }
    std::printf("replay_index_view_test: all passed\n");
    return 0;
}
