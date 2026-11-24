"""FLAT loads and stores (tools/rdna2spirv.py Gen.flat_mem).

Rule: `flat_load_*` / `flat_store_*` with a 64-bit VGPR address are the GLOBAL access with `off` (the RDNA2 ISA's
flat-memory chapter: an address outside the LDS and private apertures reads and writes global memory). The aperture
bases come from src_shared_base / src_private_base, which the translator does not accept, so no translated program
forms an aperture address from them. A non-zero instruction offset is refused (not measured on the console's GPU;
LLVM marks GFX10.1 with FeatureFlatSegmentOffsetBug and folds no offset into a FLAT access there), as are the FLAT
atomics and a 32-bit address.

The programs are hand-written. Each one is translated, run on 32 lanes with spirv_mini_interp, and its memory is
compared with words worked out by hand from the little-endian seed, and with the same program spelled with the
GLOBAL instruction and `off`.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_global_sub_dword import BASE, OUT, rdna2spirv, run, si, translate  # noqa: E402  (also puts tools/ on sys.path)
from test_global_atomics import run as run_lanes  # noqa: E402  (v0 = lane index)
from shader_ir import parse_line  # noqa: E402

M32 = 0xFFFFFFFF
# Bytes 0x11 0x22 ... 0xCC at BASE..BASE+11; the third word's top byte has its sign bit set.
SEED = {BASE: 0x44332211, BASE + 4: 0x88776655, BASE + 8: 0xCCBBAA99}


def addr(a, hi=0):
    """v[2:3] = the 64-bit address a + (hi << 32)."""
    return [f"v_mov_b32_e32 v2, {a:#x}", f"v_mov_b32_e32 v3, {hi:#x}"]


def to_global(body):
    """The same program with every flat_<x> v.., v[2:3] spelled global_<x> v.., v[2:3], off."""
    out = []
    for line in body:
        if line.startswith("flat_"):
            out.append("global_" + line[len("flat_"):] + ", off")
        else:
            out.append(line)
    return out


def out_store(reg="v5"):
    return [f"v_mov_b32_e32 v6, {OUT:#x}", f"global_store_dword v6, {reg}, s[0:1]"]


class FlatLoadStore(unittest.TestCase):

    def check(self, body, want, memory=None):
        """Run body with flat ops, assert the words in `want`, and that the global/off spelling leaves the same memory."""
        seed = dict(SEED if memory is None else memory)
        flat = run(body, memory=seed)
        for a, w in want.items():
            self.assertEqual(flat.get(a), w, f"word at {a:#x}")
        self.assertEqual(flat, run(to_global(body), memory=seed))
        return flat

    def test_dword_loads(self):
        for n in (1, 2, 3, 4):
            op = "flat_load_dword" + (f"x{n}" if n > 1 else "")
            dst = "v10" if n == 1 else f"v[10:{9 + n}]"
            with self.subTest(op=op):
                body = addr(BASE) + [f"{op} {dst}, v[2:3]", "s_waitcnt vmcnt(0) lgkmcnt(0)",
                                     f"v_mov_b32_e32 v6, {OUT:#x}"]
                body += [f"global_store_dword v6, v{10 + k}, s[0:1] offset:{4 * k}" for k in range(n)]
                want = {OUT + 4 * k: SEED.get(BASE + 4 * k, 0) for k in range(n)}
                self.check(body, want)

    def test_dword_stores(self):
        vals = [0xA1B2C3D4, 0x0BADC0DE, 0xFEEDFACE, 0x12345678]
        for n in (1, 2, 3, 4):
            op = "flat_store_dword" + (f"x{n}" if n > 1 else "")
            src = "v10" if n == 1 else f"v[10:{9 + n}]"
            with self.subTest(op=op):
                body = addr(BASE + 0x100) + [f"v_mov_b32_e32 v{10 + k}, {vals[k]:#x}" for k in range(n)]
                body.append(f"{op} v[2:3], {src}")
                self.check(body, {BASE + 0x100 + 4 * k: vals[k] for k in range(n)})

    def test_sub_dword_loads(self):
        # Byte b of the seed is 0x11 * (b + 1) for b < 12: ubyte at BASE+2 is 0x33, sbyte at BASE+11 is 0xCC -> -52.
        cases = [("flat_load_ubyte", 2, 0x33), ("flat_load_ubyte", 11, 0xCC), ("flat_load_sbyte", 11, 0xFFFFFFCC),
                 ("flat_load_sbyte", 5, 0x66), ("flat_load_ushort", 0, 0x2211), ("flat_load_ushort", 3, 0x5544),
                 ("flat_load_sshort", 10, 0xFFFFCCBB), ("flat_load_sshort", 7, 0xFFFF9988)]
        for op, off, want in cases:
            with self.subTest(op=op, off=off):
                body = addr(BASE + off) + [f"{op} v5, v[2:3]", "s_waitcnt vmcnt(0) lgkmcnt(0)"] + out_store()
                self.check(body, {OUT: want})

    def test_d16_loads_keep_the_other_half(self):
        cases = [("flat_load_ubyte_d16", 1, 0xDEAD0022), ("flat_load_ubyte_d16_hi", 1, 0x0022BEEF),
                 ("flat_load_sbyte_d16", 11, 0xDEADFFCC), ("flat_load_sbyte_d16_hi", 11, 0xFFCCBEEF),
                 ("flat_load_short_d16", 3, 0xDEAD5544), ("flat_load_short_d16_hi", 4, 0x6655BEEF)]
        for op, off, want in cases:
            with self.subTest(op=op, off=off):
                body = addr(BASE + off) + ["v_mov_b32_e32 v5, 0xdeadbeef", f"{op} v5, v[2:3]",
                                           "s_waitcnt vmcnt(0) lgkmcnt(0)"] + out_store()
                self.check(body, {OUT: want})

    def test_sub_dword_stores_merge_into_the_word(self):
        cases = [("flat_store_byte", 1, {BASE: 0x4433DD11}), ("flat_store_byte_d16_hi", 2, {BASE: 0x44BB2211}),
                 ("flat_store_short", 0, {BASE: 0x4433CCDD}),
                 ("flat_store_short", 3, {BASE: 0xDD332211, BASE + 4: 0x887766CC}),
                 ("flat_store_short_d16_hi", 6, {BASE + 4: 0xAABB6655})]
        for op, off, want in cases:
            with self.subTest(op=op, off=off):
                body = addr(BASE + off) + ["v_mov_b32_e32 v5, 0xaabbccdd", f"{op} v[2:3], v5"]
                self.check(body, want)

    def test_high_address_dword_is_used(self):
        far = (1 << 32) + BASE
        body = addr(BASE, hi=1) + ["flat_load_dword v5, v[2:3]", "s_waitcnt vmcnt(0) lgkmcnt(0)"] + out_store()
        self.check(body, {OUT: 0x5A5A5A5A}, memory={**SEED, far: 0x5A5A5A5A})
        body = addr(BASE, hi=1) + ["v_mov_b32_e32 v5, 0x600dc0de", "flat_store_dword v[2:3], v5"]
        mem = self.check(body, {far: 0x600DC0DE, BASE: SEED[BASE]})
        self.assertEqual(mem[BASE], SEED[BASE])

    def test_only_active_lanes_store(self):
        # 32 lanes; lane i stores i to BASE + 0x200 + 4i, and v_cmpx keeps lanes 0..7 (8 > v0).
        body = ["v_lshlrev_b32_e32 v2, 2, v0", f"v_add_nc_u32_e32 v2, {BASE + 0x200:#x}, v2", "v_mov_b32_e32 v3, 0",
                "v_cmpx_gt_u32_e32 8, v0", "flat_store_dword v[2:3], v0"]
        mem = run_lanes(body, {}, nlanes=32)
        self.assertEqual(mem, {BASE + 0x200 + 4 * i: i for i in range(8)})
        self.assertEqual(mem, run_lanes(to_global(body), {}, nlanes=32))

    def test_zero_offset_is_accepted(self):
        body = addr(BASE + 4) + ["flat_load_dword v5, v[2:3] offset:0", "s_waitcnt vmcnt(0) lgkmcnt(0)"] + out_store()
        self.assertEqual(run(body, memory=dict(SEED))[OUT], SEED[BASE + 4])

    def test_refused_forms(self):
        for line in ("flat_load_dword v5, v[2:3] offset:4", "flat_store_dword v[2:3], v5 offset:-4",
                     "flat_load_dword v5, v[2:3] offset:zz", "flat_atomic_add v[2:3], v5",
                     "flat_atomic_cmpswap v5, v[2:3], v[6:7] glc", "flat_load_dword v5, v2"):
            with self.subTest(line=line):
                ins = parse_line(line)
                with self.assertRaises(rdna2spirv.Unsupported):
                    rdna2spirv.Gen("cs").flat_mem(ins["op"], ins["args"], ins["mods"])
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(addr(BASE) + ["v_mov_b32_e32 v5, 1", "v_mov_b32_e32 v6, 0", "v_mov_b32_e32 v7, 0",
                                            line])

    def test_aperture_base_operand_is_refused(self):
        # The only way a program builds an LDS or scratch aperture address; refusing it keeps every FLAT access global.
        for src in ("src_shared_base", "src_private_base"):
            with self.subTest(src=src):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate([f"s_mov_b32 s4, {src}", "v_mov_b32_e32 v2, 0", "v_mov_b32_e32 v3, s4",
                               "flat_store_dword v[2:3], v2"])


class FlatLiveness(unittest.TestCase):

    def test_store_reads_address_and_data(self):
        insns = [parse_line(x) for x in ("flat_store_dwordx2 v[2:3], v[4:5]", "flat_store_byte v[6:7], v8")]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(insns), {2, 3, 4, 5, 6, 7, 8})

    def test_d16_load_reads_its_destination(self):
        for op in ("flat_load_short_d16", "flat_load_ubyte_d16_hi"):
            with self.subTest(op=op):
                self.assertEqual(rdna2spirv.read_before_write_vgprs([parse_line(f"{op} v9, v[2:3]")]), {2, 3, 9})
        self.assertEqual(rdna2spirv.read_before_write_vgprs([parse_line("flat_load_dword v9, v[2:3]")]), {2, 3})


if __name__ == "__main__":
    unittest.main()
