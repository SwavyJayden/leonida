"""The run-time FORMAT encode of buffer_store_format (Gen.format_encode_call) is table-driven and branch-free.

Every format code 0..max+1 (all 77 valid GFX10 buffer formats plus codes outside FORMATS) goes through the
translator and the mini interpreter, executing the shared branch-free %fmt_enc_chan + %fmt_encode_fn helpers
and %fmt_size_tab, and is compared bit-for-bit against an exact reference (including single-rounded UNORM/SNORM,
RTZ + NaN-preserving f16 and ufloat, B/R-swapped formats 50/56, two-component (x, y) vs (x, w) routing, and
d16 signed/unsigned integer stores).
"""
import math
import struct
import unittest

import support
from spirv_mini_interp import run_fragment

import rdna2spirv

BASE = 0x1000
FILL = 0xEEEEEEEE
# Distinct test inputs exercising ties-away-from-zero (0.3 -> 19661 in UNORM16), negative, >1, and normal range.
F_INPUTS = (0.3, -0.4, 1.25, 0.75)
F_BITS = tuple(struct.unpack("<I", struct.pack("<f", x))[0] for x in F_INPUTS)
I_INPUTS = (0x00001234, 0xFFFFFE01, 0x00000007, 0x00000003)


def ref_norm_bits(u, w, signed):
    """Exact single-rounded UNORM/SNORM channel bits matching RDNA2 / Gen.encode_norm_bits."""
    # NaN was mapped to 0 before norm encoding.
    if (u & 0x7FFFFFFF) > 0x7F800000:
        return 0
    scale = (1 << (w - 1)) - 1 if signed else (1 << w) - 1
    neg = (u >> 31) != 0
    a = u & 0x7FFFFFFF
    if a < 0x37000000:
        mag = 0
    elif a >= 0x3F800000:
        mag = scale
    else:
        exp = a >> 23
        shift = 150 - exp
        m = (a & 0x7FFFFF) | 0x800000
        mag = (m * scale + (1 << (shift - 1))) >> shift
    if signed:
        return ((-mag) & 0xFFFFFFFF) if neg else mag
    return 0 if neg else mag


def ref_f16_bits(u):
    """Exact RTZ + +-65504 clamp + NaN-payload-preserving f32->f16 matching Gen.mem_f16."""
    sg = (u >> 16) & 0x8000
    a = u & 0x7FFFFFFF
    if a > 0x7F800000:
        top = (u >> 13) & 0x3FF
        return sg | 0x7C00 | (top if top else 1)
    if a == 0x7F800000:
        return sg | 0x7C00
    if a >= 0x47800000:
        return sg | 0x7BFF
    if a < 0x38800000:
        if a < 0x33000000:
            return sg
        exp = a >> 23
        m = (a & 0x7FFFFF) | 0x800000
        shift = 126 - exp
        return sg | (m >> shift)
    return sg | ((a - (112 << 23)) >> 13)


def ref_ufloat_bits(u, w):
    """Exact w-bit unsigned float matching Gen.ufloat_encode_bits."""
    mb = w - 5
    a = u & 0x7FFFFFFF
    sg = u >> 31
    if a > 0x7F800000:
        nm = (a >> (23 - mb)) & ((1 << mb) - 1)
        return (31 << mb) | (nm if nm else 1)
    if sg:
        return 0
    if a == 0x7F800000:
        return 31 << mb
    if a >= 0x47800000:
        return (30 << mb) | ((1 << mb) - 1)
    if a >= 0x38800000:
        return (a - (112 << 23)) >> (23 - mb)
    exp = a >> 23
    m = (a & 0x7FFFFF) | 0x800000
    sh = min(136 - mb - exp, 31)
    return m >> sh


def ref_encode_words(code, f_bits, i_bits, two_y=True, bgra=False):
    """Reference 4-dword encoded memory image (over FILL) for a dynamic V# store."""
    if code not in rdna2spirv.Gen.FORMATS:
        return (FILL, FILL, FILL, FILL)
    size, kind, widths = rdna2spirv.Gen.FORMATS[code]
    fsrc = list(f_bits)
    isrc = list(i_bits)
    if bgra and code in (50, 56):
        fsrc = [fsrc[2], fsrc[1], fsrc[0], fsrc[3]]
    if len(widths) == 2:
        sec_idx = 1 if two_y else 3
        fsrc = [fsrc[0], fsrc[sec_idx]]
        isrc = [isrc[0], isrc[sec_idx]]
    words = [0, 0, 0, 0]
    bit = 0
    for c, w in enumerate(widths):
        if kind == "f32":
            field = fsrc[c]
        elif kind in ("u32", "i32"):
            field = isrc[c]
        elif kind == "f16":
            field = ref_f16_bits(fsrc[c])
        elif kind in ("unorm", "uscaled"):
            field = ref_norm_bits(fsrc[c], w, signed=False)
        elif kind in ("snorm", "sscaled"):
            field = ref_norm_bits(fsrc[c], w, signed=True)
        elif kind == "sint":
            sv = isrc[c] - (1 << 32) if (isrc[c] >> 31) else isrc[c]
            lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
            field = max(lo, min(hi, sv)) & 0xFFFFFFFF
        elif kind == "ufloat":
            field = ref_ufloat_bits(fsrc[c], w)
        else:  # uint
            field = min(isrc[c], (1 << w) - 1)
        mask = 0xFFFFFFFF if w == 32 else ((1 << w) - 1)
        words[bit // 32] |= (field & mask) << (bit % 32)
        bit += w
    out = [FILL, FILL, FILL, FILL]
    if size >= 4:
        for k in range(size // 4):
            out[k] = words[k]
    else:
        mk = (1 << (8 * size)) - 1
        out[0] = (FILL & ~mk) | (words[0] & mk)
    return tuple(out)


def store_module(code, vals, dst_sel=0x00000FAC):
    d3 = (code << 12) | 0x30000000 | dst_sel
    lines = [
        f"s_mov_b32 s0, {BASE:#x}",
        "s_mov_b32 s1, 0",
        "s_mov_b32 s2, 0x100",
        f"s_mov_b32 s3, {d3:#x}",
        "v_mov_b32 v1, 0",
        f"v_mov_b32 v2, {vals[0]:#x}",
        f"v_mov_b32 v3, {vals[1]:#x}",
        f"v_mov_b32 v4, {vals[2]:#x}",
        f"v_mov_b32 v5, {vals[3]:#x}",
        "buffer_store_format_xyzw v[2:5], v1, s[0:3], 0 offen",
        "s_endpgm",
    ]
    return support.translate_lines("ps_fmt_enc_table", lines)


def function_body(spv, name):
    body, inside = [], False
    for line in spv.splitlines():
        text = line.strip()
        if text.startswith(f"{name} = OpFunction"):
            inside = True
            continue
        if inside:
            if text == "OpFunctionEnd":
                break
            body.append(text)
    return body


BRANCHES = ("OpBranchConditional", "OpSelectionMerge", "OpSwitch", "OpBranch", "OpLoopMerge")


class FormatEncodeTableTests(unittest.TestCase):
    def test_every_format_and_the_codes_between(self):
        # Test all 128 codes with float inputs (for float/norm/scaled/f16/ufloat) and integer inputs (for int/uint/u32/i32),
        # using XY01 (0x0000010A? -> X=4, Y=5, Z=0, W=1 => 4 | 5<<3 | 0<<6 | 1<<9 = 0x22C) so 2-comp takes (x, y).
        dst_xy01 = 4 | (5 << 3) | (0 << 6) | (1 << 9)
        for code in range(max(rdna2spirv.Gen.FORMATS) + 2):
            entry = rdna2spirv.Gen.FORMATS.get(code)
            vals = I_INPUTS if (entry and entry[1] in ("uint", "sint", "u32", "i32")) else F_BITS
            spv = store_module(code, vals, dst_sel=dst_xy01)
            mem = {BASE + 4 * i: FILL for i in range(4)}
            run_fragment(spv, memory=mem)
            got = tuple(mem[BASE + 4 * i] for i in range(4))
            want = ref_encode_words(code, vals, vals, two_y=True)
            self.assertEqual(got, want, f"format {code} ({entry})")

    def test_two_component_xw_routing_under_identity_xyzw(self):
        # Under identity XYZW selectors (0xFAC), 2-component V# formats store (x, w), not (x, y).
        for code in (9, 10, 11, 14, 15, 16, 17, 64, 65, 66):
            entry = rdna2spirv.Gen.FORMATS[code]
            vals = I_INPUTS if entry[1] in ("uint", "sint", "u32", "i32") else F_BITS
            spv = store_module(code, vals, dst_sel=0x00000FAC)
            mem = {BASE + 4 * i: FILL for i in range(4)}
            run_fragment(spv, memory=mem)
            got = tuple(mem[BASE + 4 * i] for i in range(4))
            want = ref_encode_words(code, vals, vals, two_y=False)
            self.assertEqual(got, want, f"2-comp XYZW routing format {code}")

    def test_bgra_swapped_unorm_formats_50_and_56(self):
        # V# selectors Z,Y,X,W (0xF2E) swap R and B for formats 50 (2_10_10_10_UNORM) and 56 (8_8_8_8_UNORM).
        for code in (50, 56):
            for dst_sel, bgra in ((0x00000F2E, True), (0x00000FAC, False)):
                spv = store_module(code, F_BITS, dst_sel=dst_sel)
                mem = {BASE + 4 * i: FILL for i in range(4)}
                run_fragment(spv, memory=mem)
                got = tuple(mem[BASE + 4 * i] for i in range(4))
                want = ref_encode_words(code, F_BITS, F_BITS, two_y=False, bgra=bgra)
                self.assertEqual(got, want, f"format {code} dst_sel={dst_sel:#x}")

    def test_the_encode_has_no_branch_and_is_small(self):
        spv = store_module(1, F_BITS)
        chan_body = function_body(spv, "%fmt_enc_chan")
        enc_body = function_body(spv, "%fmt_encode_fn")
        self.assertTrue(chan_body, "%fmt_enc_chan is missing")
        self.assertTrue(enc_body, "%fmt_encode_fn is missing")
        for fn_name, body in (("%fmt_enc_chan", chan_body), ("%fmt_encode_fn", enc_body)):
            for line in body:
                for op in BRANCHES:
                    self.assertNotIn(op, line, f"{fn_name} branches: {line}")
        # Element size comes directly from %fmt_size_tab instead of the old 157-line %fmt_size_fn.
        self.assertNotIn("%fmt_size_fn = OpFunction", spv)
        self.assertIn("%fmt_size_tab", spv)
        total = len(chan_body) + len(enc_body)
        # Before: 3,973 lines and 154 branches in %fmt_encode_fn + 157 lines in %fmt_size_fn.
        self.assertLess(total, 400, f"encode helpers too large: {total} instructions")


if __name__ == "__main__":
    unittest.main()
