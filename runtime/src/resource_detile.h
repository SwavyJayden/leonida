#pragma once

// Detiling and tiling for AMD GFX10 / RDNA2 console-tiled resources (docs/d3d12_vulkan.md section 4.2).
//
// Resources created through Xbox slot 76 carry layout 0x101 (kXboxTiledLayout).
// The data layout inside each 64 KiB tile is GFX10 SW_64KB_D (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO):
// 64 KiB macro-blocks are laid out in row-major order across the surface, and intra-tile coordinates (x, y)
// are swizzled into byte offsets via bit-permutation equations matching RDNA2 / Xbox Series hardware.
//
// Independent X and Y coordinate bit mappings permit O(1) intra-tile offset resolution via compile-time
// lookup tables (lut_x[x % tile_w] | lut_y[y % tile_h]). No Windows or Vulkan dependency.

#include "resource_tiling.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <span>

namespace leonida {

// The Xbox tiled layout word (device slot 76, docs/d3d12_vulkan.md section 4.2).
inline constexpr std::uint32_t xbox_tiled_layout = 0x101;

constexpr bool is_xbox_tiled(std::uint32_t layout) {
    return layout == xbox_tiled_layout;
}

// Intra-tile bit permutation equations for 64 KiB tiles (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO).
// For each bit 0..15:
//   s > 0: bit is ((x >> (s - 1)) & 1)
//   s < 0: bit is ((y >> (-s - 1)) & 1)
//   s == 0: intra-element byte bit (element_bytes > 1)
template <std::uint32_t ElementBytes>
struct SwizzleLut64K {
    static constexpr std::uint32_t bpe = ElementBytes;
    static constexpr TileShape shape = element_tile_shape(false, bpe);
    static constexpr std::uint32_t tile_w = shape.width;
    static constexpr std::uint32_t tile_h = shape.height;

    std::array<std::uint16_t, tile_w> lut_x{};
    std::array<std::uint16_t, tile_h> lut_y{};

    constexpr SwizzleLut64K() {
        constexpr std::array<int8_t, 16> pat = []() {
            if constexpr (bpe == 1) {
                return std::array<int8_t, 16>{1, 2, 3, -2, -1, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8};
            } else if constexpr (bpe == 2) {
                return std::array<int8_t, 16>{0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8};
            } else if constexpr (bpe == 4) {
                // RDNA2/RbPlus SW_64KB_D: bits 6..7 swapped vs Navi 1x ((3, -3) instead of (-3, 3))
                return std::array<int8_t, 16>{0, 0, 1, 2, -1, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7};
            } else if constexpr (bpe == 8) {
                return std::array<int8_t, 16>{0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7};
            } else if constexpr (bpe == 16) {
                return std::array<int8_t, 16>{0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6};
            } else {
                return std::array<int8_t, 16>{};
            }
        }();

        for (std::uint32_t x = 0; x < tile_w; ++x) {
            std::uint16_t val = 0;
            for (std::uint32_t bit = 0; bit < 16; ++bit) {
                const int8_t s = pat[bit];
                if (s > 0) {
                    val |= static_cast<std::uint16_t>(((x >> (s - 1)) & 1) << bit);
                }
            }
            lut_x[x] = val;
        }

        for (std::uint32_t y = 0; y < tile_h; ++y) {
            std::uint16_t val = 0;
            for (std::uint32_t bit = 0; bit < 16; ++bit) {
                const int8_t s = pat[bit];
                if (s < 0) {
                    val |= static_cast<std::uint16_t>(((y >> (-s - 1)) & 1) << bit);
                }
            }
            lut_y[y] = val;
        }
    }
};

inline constexpr SwizzleLut64K<1> swizzle_lut_64k_1bpe{};
inline constexpr SwizzleLut64K<2> swizzle_lut_64k_2bpe{};
inline constexpr SwizzleLut64K<4> swizzle_lut_64k_4bpe{};
inline constexpr SwizzleLut64K<8> swizzle_lut_64k_8bpe{};
inline constexpr SwizzleLut64K<16> swizzle_lut_64k_16bpe{};

namespace detail {

constexpr bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
    if (a != 0 && b > (std::numeric_limits<std::size_t>::max)() / a) return false;
    out = a * b;
    return true;
}

struct SurfaceLayout64K {
    std::uint32_t blocks_w = 0;
    std::uint32_t blocks_h = 0;
    std::uint32_t tiles_x = 0;
    std::uint32_t tiles_y = 0;
    std::size_t linear_bytes = 0;
    std::size_t tiled_bytes = 0;
};

constexpr bool compute_surface_layout_64k(std::uint32_t width, std::uint32_t height,
                                          std::uint32_t element_bytes, std::uint32_t block_dim,
                                          SurfaceLayout64K& out) {
    if (!width || !height || (block_dim != 1 && block_dim != 4)) return false;
    const TileShape shape = element_tile_shape(false, element_bytes);
    if (!shape.width || !shape.height) return false;

    // Overflow-free ceiling division for positive uint32_t operands: 1 + (n - 1) / d.
    const std::uint32_t blocks_w = 1u + (width - 1u) / block_dim;
    const std::uint32_t blocks_h = 1u + (height - 1u) / block_dim;
    const std::uint32_t tiles_x = 1u + (blocks_w - 1u) / shape.width;
    const std::uint32_t tiles_y = 1u + (blocks_h - 1u) / shape.height;

    std::size_t block_count = 0;
    std::size_t linear_bytes = 0;
    std::size_t tile_count = 0;
    std::size_t tiled_bytes = 0;
    if (!checked_mul(blocks_w, blocks_h, block_count) ||
        !checked_mul(block_count, element_bytes, linear_bytes) ||
        !checked_mul(tiles_x, tiles_y, tile_count) ||
        !checked_mul(tile_count, tile_size_in_bytes, tiled_bytes)) {
        return false;
    }

    out = {blocks_w, blocks_h, tiles_x, tiles_y, linear_bytes, tiled_bytes};
    return true;
}

inline bool spans_overlap(const std::byte* a, std::size_t a_len,
                          const std::byte* b, std::size_t b_len) {
    if (!a_len || !b_len) return false;
    const std::less<const std::byte*> lt{};
    return lt(a, b + b_len) && lt(b, a + a_len);
}

}  // namespace detail

// Compute required tiled byte size for a 2D surface of the given dimensions (0 on invalid input or overflow).
constexpr std::size_t tiled_surface_size_64k(std::uint32_t width, std::uint32_t height,
                                            std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    return detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)
               ? layout.tiled_bytes
               : 0;
}

// Compute required linear byte size for a 2D surface of the given dimensions (0 on invalid input or overflow).
constexpr std::size_t linear_surface_size(std::uint32_t width, std::uint32_t height,
                                         std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    return detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)
               ? layout.linear_bytes
               : 0;
}

namespace detail {

template <std::uint32_t Bpe>
constexpr const auto& swizzle_lut_64k() {
    if constexpr (Bpe == 1) return swizzle_lut_64k_1bpe;
    else if constexpr (Bpe == 2) return swizzle_lut_64k_2bpe;
    else if constexpr (Bpe == 4) return swizzle_lut_64k_4bpe;
    else if constexpr (Bpe == 8) return swizzle_lut_64k_8bpe;
    else if constexpr (Bpe == 16) return swizzle_lut_64k_16bpe;
}

// Micro-tile geometry inside GFX10 SW_64KB_D (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO).
// Every BPE maps an aligned micro_w x micro_h block into contiguous cache lines in tiled memory:
//   -  1 BPE: 8x8  =  64 B (1 cache line; rows in tiled order 0,2,1,3,4,6,5,7 x 8 B)
//   -  2 BPE: 8x4  =  64 B (1 cache line; 4 contiguous rows x 16 B)
//   -  4 BPE: 4x4  =  64 B (1 cache line; 4 contiguous rows x 16 B)
//   -  8 BPE: 4x2  =  64 B (1 cache line; two 2x2 halves of 16 B rows at +0/+16 and +32/+48)
//   - 16 BPE: 4x4  = 256 B (4 consecutive cache lines; 4 linear rows of 64 B each)
template <std::uint32_t Bpe>
struct MicroTile64K {
    static constexpr std::uint32_t width = (Bpe == 1) ? 8u : (Bpe == 2) ? 8u : (Bpe == 4) ? 4u : (Bpe == 8) ? 4u : 4u;
    static constexpr std::uint32_t height = (Bpe == 1) ? 8u : (Bpe == 2) ? 4u : (Bpe == 4) ? 4u : (Bpe == 8) ? 2u : 4u;
    static constexpr std::uint32_t run_width = (Bpe == 1) ? 8u : (Bpe == 2) ? 8u : (Bpe == 4) ? 4u : 2u;
    static constexpr std::uint32_t bytes = width * height * Bpe;
};

// Compile-time proof that the micro-tile and contiguous-run offsets match SwizzleLut64K<Bpe>
// across the entire 64 KiB tile (disjoint X/Y bit masks, exact intra-micro-tile offsets).
template <std::uint32_t Bpe>
constexpr bool verify_microtile_layout_64k() {
    constexpr const auto& lut = swizzle_lut_64k<Bpe>();
    constexpr std::uint32_t tile_w = lut.tile_w;
    constexpr std::uint32_t tile_h = lut.tile_h;
    constexpr std::uint32_t mw = MicroTile64K<Bpe>::width;
    constexpr std::uint32_t mh = MicroTile64K<Bpe>::height;
    constexpr std::uint32_t rw = MicroTile64K<Bpe>::run_width;
    constexpr std::uint16_t low_mask = static_cast<std::uint16_t>(MicroTile64K<Bpe>::bytes - 1u);

    if (tile_w % mw != 0 || tile_h % mh != 0 || tile_w % rw != 0) return false;

    // 1. Disjointness of X and Y swizzle bits across the full tile guarantees (lut_x | lut_y) == (lut_x + lut_y).
    std::uint16_t or_x = 0;
    std::uint16_t or_y = 0;
    for (std::uint32_t x = 0; x < tile_w; ++x) or_x = static_cast<std::uint16_t>(or_x | lut.lut_x[x]);
    for (std::uint32_t y = 0; y < tile_h; ++y) or_y = static_cast<std::uint16_t>(or_y | lut.lut_y[y]);
    if ((or_x & or_y) != 0) return false;

    // 2. X offsets within every aligned micro_w slice (and every aligned run_width slice).
    for (std::uint32_t x = 0; x < tile_w; x += mw) {
        const std::uint16_t bx = lut.lut_x[x];
        if ((bx & low_mask) != 0) return false;
        for (std::uint32_t dx = 0; dx < mw; ++dx) {
            std::uint32_t expected_dx = 0;
            if constexpr (Bpe == 1 || Bpe == 2 || Bpe == 4) {
                expected_dx = dx * Bpe;
            } else if constexpr (Bpe == 8) {
                expected_dx = (dx >= 2u ? 32u : 0u) + (dx & 1u) * 8u;
            } else if constexpr (Bpe == 16) {
                expected_dx = (dx >= 2u ? 64u : 0u) + (dx & 1u) * 16u;
            }
            if (lut.lut_x[x + dx] != static_cast<std::uint16_t>(bx + expected_dx)) return false;
        }
    }
    for (std::uint32_t x = 0; x < tile_w; x += rw) {
        const std::uint16_t bx = lut.lut_x[x];
        for (std::uint32_t dx = 0; dx < rw; ++dx) {
            if (lut.lut_x[x + dx] != static_cast<std::uint16_t>(bx + dx * Bpe)) return false;
        }
    }

    // 3. Y offsets within every aligned micro_h slice.
    for (std::uint32_t y = 0; y < tile_h; y += mh) {
        const std::uint16_t by = lut.lut_y[y];
        if ((by & low_mask) != 0) return false;
        for (std::uint32_t dy = 0; dy < mh; ++dy) {
            std::uint32_t expected_dy = 0;
            if constexpr (Bpe == 1) {
                constexpr std::uint32_t row_off[8] = {0u, 16u, 8u, 24u, 32u, 48u, 40u, 56u};
                expected_dy = row_off[dy];
            } else if constexpr (Bpe == 2 || Bpe == 4) {
                expected_dy = dy * 16u;
            } else if constexpr (Bpe == 8) {
                expected_dy = dy * 16u;
            } else if constexpr (Bpe == 16) {
                expected_dy = (dy >= 2u ? 128u : 0u) + (dy & 1u) * 32u;
            }
            if (lut.lut_y[y + dy] != static_cast<std::uint16_t>(by + expected_dy)) return false;
        }
    }
    return true;
}

static_assert(verify_microtile_layout_64k<1>(), "1 BPE micro-tile layout mismatch");
static_assert(verify_microtile_layout_64k<2>(), "2 BPE micro-tile layout mismatch");
static_assert(verify_microtile_layout_64k<4>(), "4 BPE micro-tile layout mismatch");
static_assert(verify_microtile_layout_64k<8>(), "8 BPE micro-tile layout mismatch");
static_assert(verify_microtile_layout_64k<16>(), "16 BPE micro-tile layout mismatch");

// Strip-mined 64-byte / 256-byte cache-line micro-tile copy between linear and 64 KiB tiled layouts.
// Sweeping y in strips of micro_h rows and streaming x sequentially across macro-tiles keeps each tiled
// cache line touched completely in one pass while avoiding 2^k linear-stride L1 set conflicts.
template <std::uint32_t Bpe, bool ToTiled>
void copy_surface_64k_typed(const std::byte* src, std::byte* dst,
                            std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    constexpr const auto& lut = swizzle_lut_64k<Bpe>();
    constexpr std::uint32_t tile_w = lut.tile_w;
    constexpr std::uint32_t tile_h = lut.tile_h;
    constexpr std::uint32_t micro_w = MicroTile64K<Bpe>::width;
    constexpr std::uint32_t micro_h = MicroTile64K<Bpe>::height;
    constexpr std::uint32_t run_w = MicroTile64K<Bpe>::run_width;
    constexpr std::size_t run_bytes = std::size_t{run_w} * Bpe;

    const std::size_t stride = std::size_t{blocks_w} * Bpe;
    const std::uint32_t full_h = blocks_h & ~(micro_h - 1u);
    const std::uint32_t full_w = blocks_w & ~(micro_w - 1u);

    // Interior micro-tile strips: full_h x full_w.
    for (std::uint32_t y = 0; y < full_h; y += micro_h) {
        const std::uint32_t ty = y / tile_h;
        const std::uint32_t ly = y & (tile_h - 1u);
        const std::uint16_t base_y = lut.lut_y[ly];
        const std::size_t macro_y_base = std::size_t{ty} * tiles_x * tile_size_in_bytes;
        const std::size_t lin_y_base = std::size_t{y} * stride;

        std::uint32_t x = 0;
        for (std::uint32_t tx = 0; tx < tiles_x && x < full_w; ++tx) {
            const std::size_t tile_base = macro_y_base + std::size_t{tx} * tile_size_in_bytes + base_y;
            const std::size_t lin_tile_base = lin_y_base + std::size_t{x} * Bpe;
            const std::uint32_t tile_cols = (std::min)(tile_w, full_w - x);

            for (std::uint32_t lx = 0; lx < tile_cols; lx += micro_w) {
                const std::size_t t0 = tile_base + lut.lut_x[lx];
                const std::size_t l0 = lin_tile_base + std::size_t{lx} * Bpe;
                if constexpr (Bpe == 1) {
                    if constexpr (ToTiled) {
                        std::memcpy(dst + t0 + 0u,  src + l0 + 0u * stride, 8u);
                        std::memcpy(dst + t0 + 8u,  src + l0 + 2u * stride, 8u);
                        std::memcpy(dst + t0 + 16u, src + l0 + 1u * stride, 8u);
                        std::memcpy(dst + t0 + 24u, src + l0 + 3u * stride, 8u);
                        std::memcpy(dst + t0 + 32u, src + l0 + 4u * stride, 8u);
                        std::memcpy(dst + t0 + 40u, src + l0 + 6u * stride, 8u);
                        std::memcpy(dst + t0 + 48u, src + l0 + 5u * stride, 8u);
                        std::memcpy(dst + t0 + 56u, src + l0 + 7u * stride, 8u);
                    } else {
                        std::memcpy(dst + l0 + 0u * stride, src + t0 + 0u,  8u);
                        std::memcpy(dst + l0 + 1u * stride, src + t0 + 16u, 8u);
                        std::memcpy(dst + l0 + 2u * stride, src + t0 + 8u,  8u);
                        std::memcpy(dst + l0 + 3u * stride, src + t0 + 24u, 8u);
                        std::memcpy(dst + l0 + 4u * stride, src + t0 + 32u, 8u);
                        std::memcpy(dst + l0 + 5u * stride, src + t0 + 48u, 8u);
                        std::memcpy(dst + l0 + 6u * stride, src + t0 + 40u, 8u);
                        std::memcpy(dst + l0 + 7u * stride, src + t0 + 56u, 8u);
                    }
                } else if constexpr (Bpe == 2 || Bpe == 4) {
                    if constexpr (ToTiled) {
                        std::memcpy(dst + t0 + 0u,  src + l0 + 0u * stride, 16u);
                        std::memcpy(dst + t0 + 16u, src + l0 + 1u * stride, 16u);
                        std::memcpy(dst + t0 + 32u, src + l0 + 2u * stride, 16u);
                        std::memcpy(dst + t0 + 48u, src + l0 + 3u * stride, 16u);
                    } else {
                        std::memcpy(dst + l0 + 0u * stride, src + t0 + 0u,  16u);
                        std::memcpy(dst + l0 + 1u * stride, src + t0 + 16u, 16u);
                        std::memcpy(dst + l0 + 2u * stride, src + t0 + 32u, 16u);
                        std::memcpy(dst + l0 + 3u * stride, src + t0 + 48u, 16u);
                    }
                } else if constexpr (Bpe == 8) {
                    if constexpr (ToTiled) {
                        std::memcpy(dst + t0 + 0u,  src + l0,               16u);
                        std::memcpy(dst + t0 + 16u, src + l0 + stride,      16u);
                        std::memcpy(dst + t0 + 32u, src + l0 + 16u,         16u);
                        std::memcpy(dst + t0 + 48u, src + l0 + stride + 16u, 16u);
                    } else {
                        std::memcpy(dst + l0,               src + t0 + 0u,  16u);
                        std::memcpy(dst + l0 + 16u,         src + t0 + 32u, 16u);
                        std::memcpy(dst + l0 + stride,      src + t0 + 16u, 16u);
                        std::memcpy(dst + l0 + stride + 16u, src + t0 + 48u, 16u);
                    }
                } else if constexpr (Bpe == 16) {
                    if constexpr (ToTiled) {
                        std::memcpy(dst + t0 + 0u,   src + l0 + 0u * stride,       32u);
                        std::memcpy(dst + t0 + 32u,  src + l0 + 1u * stride,       32u);
                        std::memcpy(dst + t0 + 64u,  src + l0 + 0u * stride + 32u, 32u);
                        std::memcpy(dst + t0 + 96u,  src + l0 + 1u * stride + 32u, 32u);
                        std::memcpy(dst + t0 + 128u, src + l0 + 2u * stride,       32u);
                        std::memcpy(dst + t0 + 160u, src + l0 + 3u * stride,       32u);
                        std::memcpy(dst + t0 + 192u, src + l0 + 2u * stride + 32u, 32u);
                        std::memcpy(dst + t0 + 224u, src + l0 + 3u * stride + 32u, 32u);
                    } else {
                        std::memcpy(dst + l0 + 0u * stride,       src + t0 + 0u,   32u);
                        std::memcpy(dst + l0 + 0u * stride + 32u, src + t0 + 64u,  32u);
                        std::memcpy(dst + l0 + 1u * stride,       src + t0 + 32u,  32u);
                        std::memcpy(dst + l0 + 1u * stride + 32u, src + t0 + 96u,  32u);
                        std::memcpy(dst + l0 + 2u * stride,       src + t0 + 128u, 32u);
                        std::memcpy(dst + l0 + 2u * stride + 32u, src + t0 + 192u, 32u);
                        std::memcpy(dst + l0 + 3u * stride,       src + t0 + 160u, 32u);
                        std::memcpy(dst + l0 + 3u * stride + 32u, src + t0 + 224u, 32u);
                    }
                }
            }
            x += tile_cols;
        }

        // Right-edge scalar tail for this strip when blocks_w is not a multiple of micro_w.
        if (full_w < blocks_w) {
            for (std::uint32_t r = 0; r < micro_h; ++r) {
                const std::uint32_t yy = y + r;
                const std::uint16_t iy = lut.lut_y[yy & (tile_h - 1u)];
                const std::size_t lin_row = std::size_t{yy} * stride;
                for (std::uint32_t xx = full_w; xx < blocks_w; ++xx) {
                    const std::size_t t_off = macro_y_base + std::size_t{xx / tile_w} * tile_size_in_bytes +
                                              (lut.lut_x[xx & (tile_w - 1u)] | iy);
                    const std::size_t l_off = lin_row + std::size_t{xx} * Bpe;
                    if constexpr (ToTiled) std::memcpy(dst + t_off, src + l_off, Bpe);
                    else std::memcpy(dst + l_off, src + t_off, Bpe);
                }
            }
        }
    }

    // Bottom-edge tail when blocks_h is not a multiple of micro_h.
    const std::uint32_t run_end = blocks_w & ~(run_w - 1u);
    for (std::uint32_t y = full_h; y < blocks_h; ++y) {
        const std::uint32_t ty = y / tile_h;
        const std::uint16_t iy = lut.lut_y[y & (tile_h - 1u)];
        const std::size_t macro_y_base = std::size_t{ty} * tiles_x * tile_size_in_bytes;
        const std::size_t lin_row = std::size_t{y} * stride;
        std::uint32_t x = 0;
        for (; x < run_end; x += run_w) {
            const std::size_t t_off = macro_y_base + std::size_t{x / tile_w} * tile_size_in_bytes +
                                      (lut.lut_x[x & (tile_w - 1u)] | iy);
            const std::size_t l_off = lin_row + std::size_t{x} * Bpe;
            if constexpr (ToTiled) std::memcpy(dst + t_off, src + l_off, run_bytes);
            else std::memcpy(dst + l_off, src + t_off, run_bytes);
        }
        for (; x < blocks_w; ++x) {
            const std::size_t t_off = macro_y_base + std::size_t{x / tile_w} * tile_size_in_bytes +
                                      (lut.lut_x[x & (tile_w - 1u)] | iy);
            const std::size_t l_off = lin_row + std::size_t{x} * Bpe;
            if constexpr (ToTiled) std::memcpy(dst + t_off, src + l_off, Bpe);
            else std::memcpy(dst + l_off, src + t_off, Bpe);
        }
    }
}

template <std::uint32_t Bpe>
void detile_typed(std::span<const std::byte> tiled, std::span<std::byte> linear,
                  std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    copy_surface_64k_typed<Bpe, false>(tiled.data(), linear.data(), blocks_w, blocks_h, tiles_x);
}

template <std::uint32_t Bpe>
void tile_typed(std::span<const std::byte> linear, std::span<std::byte> tiled,
                std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    copy_surface_64k_typed<Bpe, true>(linear.data(), tiled.data(), blocks_w, blocks_h, tiles_x);
}

}  // namespace detail

// Detile a 2D surface from console-tiled 64 KiB layout into linear row-major order.
inline bool detile_surface_64k(std::span<const std::byte> tiled, std::span<std::byte> linear,
                               std::uint32_t width, std::uint32_t height,
                               std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    if (!detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)) return false;
    if (linear.size() < layout.linear_bytes || tiled.size() < layout.tiled_bytes) return false;
    if (detail::spans_overlap(tiled.data(), layout.tiled_bytes, linear.data(), layout.linear_bytes)) return false;

    switch (element_bytes) {
    case 1:
        detail::detile_typed<1>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 2:
        detail::detile_typed<2>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 4:
        detail::detile_typed<4>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 8:
        detail::detile_typed<8>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 16:
        detail::detile_typed<16>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    default:
        return false;
    }
}

// Tile a 2D surface from linear row-major order into console-tiled 64 KiB layout.
inline bool tile_surface_64k(std::span<const std::byte> linear, std::span<std::byte> tiled,
                             std::uint32_t width, std::uint32_t height,
                             std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    if (!detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)) return false;
    if (linear.size() < layout.linear_bytes || tiled.size() < layout.tiled_bytes) return false;
    if (detail::spans_overlap(linear.data(), layout.linear_bytes, tiled.data(), layout.tiled_bytes)) return false;

    switch (element_bytes) {
    case 1:
        detail::tile_typed<1>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 2:
        detail::tile_typed<2>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 4:
        detail::tile_typed<4>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 8:
        detail::tile_typed<8>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 16:
        detail::tile_typed<16>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    default:
        return false;
    }
}

}  // namespace leonida
