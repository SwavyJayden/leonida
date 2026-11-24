// Detiling and tiling round-trip verification (src/resource_detile.h), without a GPU.
//
// Tests bit-exact round-trip fidelity (linear -> tile -> detile == linear) across all RDNA2 64 KiB
// swizzle patterns (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO), covering 1, 2, 4, 8, and 16 bytes
// per element, sub-tile, exact-tile, multi-tile, non-power-of-two, and block-compressed layouts.

#include "../src/resource_detile.h"

#include <cstdio>
#include <cstdint>
#include <vector>
#include <unordered_set>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace {

using namespace leonida;

// Helper to fill a linear buffer with a deterministic synthetic pseudo-random pattern.
void fill_synthetic_linear(std::vector<std::byte>& buf, std::uint32_t seed) {
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<std::byte>((i * 37u + seed * 101u + 13u) & 0xFFu);
    }
}

// Test round-trip identity: linear -> tile -> detile == linear.
bool test_roundtrip(std::uint32_t width, std::uint32_t height,
                    std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    const std::size_t linear_size = linear_surface_size(width, height, element_bytes, block_dim);
    const std::size_t tiled_size = tiled_surface_size_64k(width, height, element_bytes, block_dim);
    if (!linear_size || !tiled_size) return false;

    std::vector<std::byte> original(linear_size);
    fill_synthetic_linear(original, width ^ (height << 8) ^ element_bytes);

    std::vector<std::byte> tiled(tiled_size, std::byte{0xCC});  // 0xCC poison
    std::vector<std::byte> recovered(linear_size, std::byte{0});

    // 1. Tile
    if (!tile_surface_64k(original, tiled, width, height, element_bytes, block_dim)) {
        std::fprintf(stderr, "tile_surface_64k failed for %ux%u eb=%u blk=%u\n",
                     width, height, element_bytes, block_dim);
        return false;
    }

    // 2. Detile
    if (!detile_surface_64k(tiled, recovered, width, height, element_bytes, block_dim)) {
        std::fprintf(stderr, "detile_surface_64k failed for %ux%u eb=%u blk=%u\n",
                     width, height, element_bytes, block_dim);
        return false;
    }

    // 3. Compare bitwise
    if (std::memcmp(original.data(), recovered.data(), linear_size) != 0) {
        std::fprintf(stderr, "mismatch in round-trip for %ux%u eb=%u blk=%u\n",
                     width, height, element_bytes, block_dim);
        return false;
    }

    // 4. Idempotence test: tile again from recovered -> must equal tiled active regions
    std::vector<std::byte> tiled2(tiled_size, std::byte{0xCC});
    if (!tile_surface_64k(recovered, tiled2, width, height, element_bytes, block_dim)) return false;
    if (!detile_surface_64k(tiled2, recovered, width, height, element_bytes, block_dim)) return false;
    if (std::memcmp(original.data(), recovered.data(), linear_size) != 0) return false;

    return true;
}

// Verify that intra-tile coordinates are 1-to-1 and strictly within [0, 65535].
template <std::uint32_t Bpe>
bool test_intra_tile_bijective() {
    constexpr SwizzleLut64K<Bpe> lut{};
    constexpr std::uint32_t tile_w = lut.tile_w;
    constexpr std::uint32_t tile_h = lut.tile_h;

    std::unordered_set<std::uint16_t> offsets;
    offsets.reserve(tile_w * tile_h);

    for (std::uint32_t y = 0; y < tile_h; ++y) {
        const std::uint16_t intra_y = lut.lut_y[y];
        for (std::uint32_t x = 0; x < tile_w; ++x) {
            const std::uint16_t intra_x = lut.lut_x[x];
            const std::uint16_t off = intra_x | intra_y;

            if (off >= tile_size_in_bytes) return false;
            // Intra-tile offset must be aligned to element_bytes:
            if (off % Bpe != 0) return false;

            offsets.insert(off);
        }
    }

    // Must be completely unique with zero collisions:
    return offsets.size() == (tile_w * tile_h);
}

// Independent element-by-element scalar reference against SwizzleLut64K<Bpe>, verifying both exact tiled/linear
// byte placement and that all inactive padding bytes in the tiled buffer stay untouched.
template <std::uint32_t Bpe>
bool test_exact_against_scalar(std::uint32_t width, std::uint32_t height, std::uint32_t block_dim = 1) {
    constexpr SwizzleLut64K<Bpe> lut{};
    const std::size_t linear_size = linear_surface_size(width, height, Bpe, block_dim);
    const std::size_t tiled_size = tiled_surface_size_64k(width, height, Bpe, block_dim);
    if (!linear_size || !tiled_size) return false;

    const std::uint32_t blocks_w = 1u + (width - 1u) / block_dim;
    const std::uint32_t blocks_h = 1u + (height - 1u) / block_dim;
    const std::uint32_t tiles_x = 1u + (blocks_w - 1u) / lut.tile_w;

    std::vector<std::byte> linear_in(linear_size);
    fill_synthetic_linear(linear_in, width * 17u + height * 31u + Bpe * 7u + block_dim);

    std::vector<std::byte> tiled_got(tiled_size, std::byte{0xCC});
    std::vector<std::byte> tiled_ref(tiled_size, std::byte{0xCC});
    if (!tile_surface_64k(linear_in, tiled_got, width, height, Bpe, block_dim)) return false;

    for (std::uint32_t y = 0; y < blocks_h; ++y) {
        const std::size_t macro_y = static_cast<std::size_t>(y / lut.tile_h) * tiles_x * tile_size_in_bytes;
        const std::uint16_t iy = lut.lut_y[y % lut.tile_h];
        const std::size_t lin_row = static_cast<std::size_t>(y) * blocks_w * Bpe;
        for (std::uint32_t x = 0; x < blocks_w; ++x) {
            const std::size_t t_off =
                macro_y + static_cast<std::size_t>(x / lut.tile_w) * tile_size_in_bytes + (lut.lut_x[x % lut.tile_w] | iy);
            std::memcpy(&tiled_ref[t_off], &linear_in[lin_row + static_cast<std::size_t>(x) * Bpe], Bpe);
        }
    }
    if (std::memcmp(tiled_got.data(), tiled_ref.data(), tiled_size) != 0) {
        std::fprintf(stderr, "tile_surface_64k diverged from scalar ref for %ux%u bpe=%u blk=%u\n",
                     width, height, Bpe, block_dim);
        return false;
    }

    // Fill the entire tiled buffer (including padding) with distinct bytes and verify detile_surface_64k matches
    // the scalar reference byte-for-byte.
    std::vector<std::byte> tiled_in(tiled_size);
    fill_synthetic_linear(tiled_in, width * 53u + height * 19u + Bpe * 11u + 3u);
    std::vector<std::byte> linear_got(linear_size, std::byte{0xAA});
    std::vector<std::byte> linear_ref(linear_size, std::byte{0xAA});
    if (!detile_surface_64k(tiled_in, linear_got, width, height, Bpe, block_dim)) return false;

    for (std::uint32_t y = 0; y < blocks_h; ++y) {
        const std::size_t macro_y = static_cast<std::size_t>(y / lut.tile_h) * tiles_x * tile_size_in_bytes;
        const std::uint16_t iy = lut.lut_y[y % lut.tile_h];
        const std::size_t lin_row = static_cast<std::size_t>(y) * blocks_w * Bpe;
        for (std::uint32_t x = 0; x < blocks_w; ++x) {
            const std::size_t t_off =
                macro_y + static_cast<std::size_t>(x / lut.tile_w) * tile_size_in_bytes + (lut.lut_x[x % lut.tile_w] | iy);
            std::memcpy(&linear_ref[lin_row + static_cast<std::size_t>(x) * Bpe], &tiled_in[t_off], Bpe);
        }
    }
    if (std::memcmp(linear_got.data(), linear_ref.data(), linear_size) != 0) {
        std::fprintf(stderr, "detile_surface_64k diverged from scalar ref for %ux%u bpe=%u blk=%u\n",
                     width, height, Bpe, block_dim);
        return false;
    }
    return true;
}

template <std::uint32_t Bpe>
bool test_all_edge_remainders() {
    constexpr SwizzleLut64K<Bpe> lut{};
    // Cover every remainder modulo micro-tile width/height (up to 8) and across single/multi-tile boundaries.
    const std::uint32_t dims_x[] = {
        1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u,
        lut.tile_w - 3u, lut.tile_w - 1u, lut.tile_w, lut.tile_w + 1u, lut.tile_w + 5u, lut.tile_w + 7u,
    };
    const std::uint32_t dims_y[] = {
        1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u,
        lut.tile_h - 1u, lut.tile_h, lut.tile_h + 1u, lut.tile_h + 3u,
    };
    for (std::uint32_t w : dims_x) {
        if (!test_exact_against_scalar<Bpe>(w, 11u)) return false;
        if (!test_exact_against_scalar<Bpe>(w, lut.tile_h + 5u)) return false;
    }
    for (std::uint32_t h : dims_y) {
        if (!test_exact_against_scalar<Bpe>(13u, h)) return false;
        if (!test_exact_against_scalar<Bpe>(lut.tile_w + 3u, h)) return false;
    }
    return true;
}

}  // namespace

int main() {
    // Layout recognition:
    CHECK(xbox_tiled_layout == 0x101);
    CHECK(is_xbox_tiled(0x101));
    CHECK(!is_xbox_tiled(0));
    CHECK(!is_xbox_tiled(0x1));

    // Intra-tile bijective property (no collisions, full coverage, element alignment):
    CHECK(test_intra_tile_bijective<1>());
    CHECK(test_intra_tile_bijective<2>());
    CHECK(test_intra_tile_bijective<4>());
    CHECK(test_intra_tile_bijective<8>());
    CHECK(test_intra_tile_bijective<16>());

    // Invalid argument & bounds rejection:
    {
        std::vector<std::byte> lin(1024), til(65536);
        CHECK(!detile_surface_64k(til, lin, 0, 100, 4));
        CHECK(!detile_surface_64k(til, lin, 100, 0, 4));
        CHECK(!detile_surface_64k(til, lin, 100, 100, 3));   // unsupported 3 bytes
        CHECK(!detile_surface_64k(til, lin, 100, 100, 5));   // unsupported 5 bytes
        CHECK(!detile_surface_64k(til, lin, 100, 100, 12));  // unsupported 12 bytes
        CHECK(!detile_surface_64k(til, lin, 100, 100, 4, 2)); // invalid block dim 2
        // Undersized destination:
        std::vector<std::byte> small_lin(10);
        CHECK(!detile_surface_64k(til, small_lin, 128, 128, 4));
        // Undersized source:
        std::vector<std::byte> small_til(100);
        CHECK(!detile_surface_64k(small_til, lin, 128, 128, 4));
        // linear_surface_size rejects unsupported element sizes consistently with tiled_surface_size_64k:
        CHECK(linear_surface_size(100, 100, 0) == 0);
        CHECK(linear_surface_size(100, 100, 3) == 0);
        CHECK(linear_surface_size(100, 100, 5) == 0);
        CHECK(linear_surface_size(100, 100, 12) == 0);
        // uint32_t ceiling-division wraparound must not wrap blocks_w / tiles_x to 0:
        CHECK(linear_surface_size(0xFFFFFFFDu, 1, 8, 4) == std::size_t{0x40000000u} * 8u);
        CHECK(tiled_surface_size_64k(0xFFFFFFFDu, 1, 8, 4) == std::size_t{0x00800000u} * tile_size_in_bytes);
        CHECK(tiled_surface_size_64k(0xFFFFFF80u, 1, 1, 1) == std::size_t{0x01000000u} * tile_size_in_bytes);
        CHECK(!detile_surface_64k(til, lin, 0xFFFFFFFDu, 1, 8, 4));
        CHECK(!tile_surface_64k(lin, til, 0xFFFFFFFDu, 1, 8, 4));
        CHECK(!detile_surface_64k(til, lin, 0xFFFFFF80u, 1, 1, 1));
        CHECK(!tile_surface_64k(lin, til, 0xFFFFFF80u, 1, 1, 1));
        // 64-bit size_t product overflow (tiles_x * tiles_y * 65536 > SIZE_MAX):
        CHECK(tiled_surface_size_64k(0xFFFF0000u, 0xFFFF0000u, 16, 1) == 0);
        CHECK(linear_surface_size(0xFFFF0000u, 0xFFFF0000u, 16, 1) == 0);
        CHECK(!detile_surface_64k(til, lin, 0xFFFF0000u, 0xFFFF0000u, 16, 1));
        CHECK(!tile_surface_64k(lin, til, 0xFFFF0000u, 0xFFFF0000u, 16, 1));
        // Overlapping spans are rejected (avoids std::memcpy UB), while adjacent non-overlapping subspans work:
        std::vector<std::byte> arena(tile_size_in_bytes + 16);
        auto aliased_tiled = std::span<std::byte>(arena).subspan(0, tile_size_in_bytes);
        auto aliased_linear = std::span<std::byte>(arena).subspan(tile_size_in_bytes - 4, 16);
        CHECK(!detile_surface_64k(aliased_tiled, aliased_linear, 2, 2, 4));
        CHECK(!tile_surface_64k(aliased_linear, aliased_tiled, 2, 2, 4));
        auto adjacent_linear = std::span<std::byte>(arena).subspan(tile_size_in_bytes, 16);
        fill_synthetic_linear(arena, 0x42u);
        CHECK(tile_surface_64k(adjacent_linear, aliased_tiled, 2, 2, 4));
        std::vector<std::byte> check_out(16);
        CHECK(detile_surface_64k(aliased_tiled, check_out, 2, 2, 4));
        CHECK(std::memcmp(adjacent_linear.data(), check_out.data(), 16) == 0);
    }

    // 1. Single-tile exact dimensions:
    CHECK(test_roundtrip(256, 256, 1));   // 1 BPP tile: 256x256
    CHECK(test_roundtrip(256, 128, 2));   // 2 BPP tile: 256x128
    CHECK(test_roundtrip(128, 128, 4));   // 4 BPP tile: 128x128
    CHECK(test_roundtrip(128, 64, 8));    // 8 BPP tile: 128x64
    CHECK(test_roundtrip(64, 64, 16));    // 16 BPP tile: 64x64

    // 2. Sub-tile dimensions (partial tiles / mips):
    CHECK(test_roundtrip(50, 30, 2));
    CHECK(test_roundtrip(100, 100, 4));
    CHECK(test_roundtrip(64, 64, 4));
    CHECK(test_roundtrip(16, 16, 8));
    CHECK(test_roundtrip(32, 32, 16));
    CHECK(test_roundtrip(1, 1, 4));

    // 3. Multi-tile power-of-two surfaces:
    CHECK(test_roundtrip(512, 512, 1));
    CHECK(test_roundtrip(512, 512, 2));
    CHECK(test_roundtrip(512, 512, 4));
    CHECK(test_roundtrip(1024, 1024, 4));
    CHECK(test_roundtrip(512, 256, 8));
    CHECK(test_roundtrip(256, 256, 16));

    // 4. Non-power-of-two and wide/tall rectangular surfaces:
    CHECK(test_roundtrip(300, 200, 4));
    CHECK(test_roundtrip(1920, 1080, 4));  // 1080p RGBA8
    CHECK(test_roundtrip(4096, 64, 4));    // thin and wide
    CHECK(test_roundtrip(64, 4096, 4));    // thin and tall
    CHECK(test_roundtrip(1024, 768, 8));
    CHECK(test_roundtrip(800, 600, 2));

    // 5. Block-compressed surfaces (BCn with block_dim = 4):
    // BC1 (8-byte block): 1024x1024 texels -> 256x256 blocks
    CHECK(test_roundtrip(1024, 1024, 8, 4));
    // BC7 (16-byte block): 2048x1024 texels -> 512x256 blocks
    CHECK(test_roundtrip(2048, 1024, 16, 4));
    // Non-multiple of 4 texel dimensions:
    CHECK(test_roundtrip(1001, 603, 8, 4));
    CHECK(test_roundtrip(1920, 1080, 16, 4));

    // 6. Exhaustive bit-exactness against independent scalar reference across all BPEs and micro-tile/tile edges:
    CHECK(test_all_edge_remainders<1>());
    CHECK(test_all_edge_remainders<2>());
    CHECK(test_all_edge_remainders<4>());
    CHECK(test_all_edge_remainders<8>());
    CHECK(test_all_edge_remainders<16>());
    CHECK(test_exact_against_scalar<8>(1001, 603, 4));
    CHECK(test_exact_against_scalar<16>(1920, 1080, 4));
    CHECK(test_exact_against_scalar<16>(263, 271, 4));

    std::printf("resource detile: all 64 KiB GFX10 swizzle round-trip and validation checks passed\n");
    return 0;
}
