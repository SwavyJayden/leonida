"""Semantic value and translation tests for RDNA2 ds_bpermute_b32 backward permute:
  ds_bpermute_b32 vD, vAddr, vData [offset:N]

RDNA2 ISA semantics (Section 10.4.4):
  * Backward (gather) permute within each 32-lane half:
    lane l reads vData from lane ((vAddr + offset) / 4) mod 32 of its own 32-lane half.
  * In wave32: executes over lanes 0..31.
  * In wave64: executes as two independent 32-lane half-waves:
    - Lanes 0..31 read exclusively from lanes 0..31.
    - Lanes 32..63 read exclusively from lanes 32..63.
  * Low 2 bits of (vAddr + offset) are ignored (byte address shifted by 2).
  * An active destination lane reading an inactive source lane (source lane off in EXEC) gets 0.
  * An inactive destination lane (destination lane off in EXEC) preserves its prior vD.

Executed end-to-end through tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py
across 32 lanes (wave32 on 32-wide subgroup) and 64 lanes (wave64 on both a single 64-wide
subgroup and two 32-wide subgroups). Clean-room compliant: zero game data, synthetic
hand-written listings only.
"""
from __future__ import annotations

import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si

VSHARP = [0, 0, 0x1000, 0x31016FAC]
PRIOR_VD = 99
ALL32 = 0xFFFFFFFF
ALL64 = (1 << 64) - 1


def fssm(threads: int, wave64: bool = False) -> bytes:
    h = bytearray(0x148)
    h[:4] = b"FSSM"
    if not wave64:
        struct.pack_into("<I", h, 0xD4, 0x20)  # FLAG_WAVE32 at 0xD4
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def translate_text(name: str, insns: list[str], threads: int, wave64: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        path = os.path.join(tmp, "disasm", name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n" + "".join("\t" + i + "\n" for i in insns))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(support.STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(fssm(threads, wave64=wave64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = support.rdna2spirv.translate(path)
            gc.collect()
        return spv


def reference_bpermute(
    data: list[int],
    addr: list[int],
    exec_mask: int,
    offset: int = 0,
    wave64: bool = False,
    prior_vd: int = PRIOR_VD,
) -> list[int]:
    """Independent Python oracle of RDNA2 ds_bpermute_b32 behavior."""
    lanes = 64 if wave64 else 32
    out = [prior_vd] * lanes
    for dst in range(lanes):
        dst_active = bool((exec_mask >> dst) & 1)
        if not dst_active:
            # Inactive destination lane preserves prior vD
            out[dst] = prior_vd
            continue
        half_base = 32 if (wave64 and dst >= 32) else 0
        target_in_half = (((addr[dst] + offset) >> 2) & 31)
        src = half_base + target_in_half
        src_active = bool((exec_mask >> src) & 1)
        if src_active:
            out[dst] = data[src]
        else:
            # Source lane is inactive in EXEC: reading returns 0
            out[dst] = 0
    return out


def make_program(
    addr_insns: list[str],
    exec_mask: int,
    offset: int = 0,
    wave64: bool = False,
    data_multiplier: int = 10,
    data_bias: int = 1,
) -> list[str]:
    """Generate compute shader:
    v2 = data_multiplier * lane + data_bias
    addr_insns write v3 (byte address)
    v10 = PRIOR_VD
    EXEC set to exec_mask
    ds_bpermute_b32 v10, v3, v2 [offset:N]
    s_waitcnt lgkmcnt(0)
    EXEC restored to all-active
    store out[lane] = v10
    """
    lines = [f"v_mad_u32_u24 v2, v0, {data_multiplier}, {data_bias}"]
    lines.extend(addr_insns)
    lines.append(f"v_mov_b32_e32 v10, {PRIOR_VD}")

    # Set EXEC mask
    lines.append(f"s_mov_b32 exec_lo, {exec_mask & ALL32:#x}")
    if wave64:
        lines.append(f"s_mov_b32 exec_hi, {(exec_mask >> 32) & ALL32:#x}")

    mod = f" offset:{offset}" if offset else ""
    lines.append(f"ds_bpermute_b32 v10, v3, v2{mod}")
    lines.append("s_waitcnt lgkmcnt(0)")

    # Restore EXEC to store outputs for all lanes
    lines.append("s_mov_b32 exec_lo, -1")
    if wave64:
        lines.append("s_mov_b32 exec_hi, -1")

    lines.extend([
        "v_lshlrev_b32_e32 v21, 2, v0",
        "buffer_store_dword v10, v21, s[0:3], 0 offen",
        "s_endpgm",
    ])
    return lines


def run_test(spv: str, threads: int, subgroup_size: int) -> list[int]:
    memory = si.run_workgroup(
        spv,
        threads,
        subgroup_size,
        inputs_for=lambda i: {
            "%wg_id": (0, 0, 0),
            "%local_id": (i, 0, 0),
            "%sg_lane": i % subgroup_size,
        },
        push_constants=VSHARP + [0] * 28,
        memory={},
    )
    return [memory.get(4 * k, 0) for k in range(threads)]


class DsBpermuteWave32Tests(unittest.TestCase):
    def test_identity(self):
        """Each lane reads its own data (addr = 4 * lane)."""
        addr_insns = ["v_lshlrev_b32_e32 v3, 2, v0"]
        data = [10 * i + 1 for i in range(32)]
        addr = [4 * i for i in range(32)]
        expected = reference_bpermute(data, addr, ALL32)

        p = make_program(addr_insns, ALL32)
        spv = translate_text("cs_bperm_w32_id", p, 32, wave64=False)
        got = run_test(spv, 32, 32)
        self.assertEqual(got, expected)

    def test_reverse(self):
        """Lane l reads lane (31 - l)."""
        addr_insns = [
            "v_sub_nc_u32_e32 v3, 31, v0",
            "v_lshlrev_b32_e32 v3, 2, v3",
        ]
        data = [10 * i + 1 for i in range(32)]
        addr = [4 * (31 - i) for i in range(32)]
        expected = reference_bpermute(data, addr, ALL32)

        p = make_program(addr_insns, ALL32)
        spv = translate_text("cs_bperm_w32_rev", p, 32, wave64=False)
        got = run_test(spv, 32, 32)
        self.assertEqual(got, expected)

    def test_cyclic_shift_with_offset(self):
        """Lane l reads lane ((l + 3) % 32) with hardware offset:12 (3 words)."""
        # addr = 4 * l, offset = 12 -> target = (4 * l + 12) >> 2 = l + 3
        addr_insns = ["v_lshlrev_b32_e32 v3, 2, v0"]
        data = [10 * i + 1 for i in range(32)]
        addr = [4 * i for i in range(32)]
        expected = reference_bpermute(data, addr, ALL32, offset=12)

        p = make_program(addr_insns, ALL32, offset=12)
        spv = translate_text("cs_bperm_w32_shift_off", p, 32, wave64=False)
        got = run_test(spv, 32, 32)
        self.assertEqual(got, expected)

    def test_broadcast_lane(self):
        """All lanes read lane 7."""
        addr_insns = ["v_mov_b32_e32 v3, 28"]  # 4 * 7 = 28
        data = [10 * i + 1 for i in range(32)]
        addr = [28] * 32
        expected = reference_bpermute(data, addr, ALL32)

        p = make_program(addr_insns, ALL32)
        spv = translate_text("cs_bperm_w32_bcast", p, 32, wave64=False)
        got = run_test(spv, 32, 32)
        self.assertEqual(got, expected)
        self.assertTrue(all(v == data[7] for v in got))

    def test_unaligned_byte_addresses(self):
        """Addresses with non-zero lower 2 bits (e.g. 4 * l + 3) drop bits 1:0."""
        addr_insns = [
            "v_lshlrev_b32_e32 v3, 2, v0",
            "v_or_b32_e32 v3, 3, v3",  # set low 2 bits
        ]
        data = [10 * i + 1 for i in range(32)]
        addr = [4 * i + 3 for i in range(32)]
        expected = reference_bpermute(data, addr, ALL32)

        p = make_program(addr_insns, ALL32)
        spv = translate_text("cs_bperm_w32_unalign", p, 32, wave64=False)
        got = run_test(spv, 32, 32)
        self.assertEqual(got, expected)

    def test_inactive_source_lane_reads_zero(self):
        """Active lanes reading inactive source lanes receive 0."""
        # EXEC = 0x55555555: even lanes active (0, 2, 4...), odd lanes inactive (1, 3, 5...).
        # Even lane l reads lane l + 1 (odd lane, hence inactive).
        # Inactive odd lanes keep PRIOR_VD.
        addr_insns = [
            "v_add_nc_u32_e32 v3, 1, v0",
            "v_lshlrev_b32_e32 v3, 2, v3",
        ]
        exec_mask = 0x55555555
        data = [10 * i + 1 for i in range(32)]
        addr = [4 * (i + 1) for i in range(32)]
        expected = reference_bpermute(data, addr, exec_mask)

        p = make_program(addr_insns, exec_mask)
        spv = translate_text("cs_bperm_w32_inact_src", p, 32, wave64=False)
        got = run_test(spv, 32, 32)
        self.assertEqual(got, expected)
        # Even lanes read inactive source -> 0; odd lanes inactive -> PRIOR_VD
        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(got[i], 0)
            else:
                self.assertEqual(got[i], PRIOR_VD)


class DsBpermuteWave64Tests(unittest.TestCase):
    def test_wave64_halves_independent(self):
        """Wave64 runs as two independent 32-lane halves on 64-wide and 32-wide subgroups."""
        # Lane l reads (l + 1) % 32 within its half
        addr_insns = [
            "v_and_b32_e32 v4, 31, v0",
            "v_add_nc_u32_e32 v4, 1, v4",
            "v_and_b32_e32 v4, 31, v4",
            "v_lshlrev_b32_e32 v3, 2, v4",
        ]
        data = [10 * i + 1 for i in range(64)]
        addr = [4 * (((i % 32) + 1) % 32) for i in range(64)]
        expected = reference_bpermute(data, addr, ALL64, wave64=True)

        p = make_program(addr_insns, ALL64, wave64=True)
        spv = translate_text("cs_bperm_w64_halves", p, 64, wave64=True)

        # 1. Native 64-wide subgroup (e.g. AMD)
        got64 = run_test(spv, 64, 64)
        self.assertEqual(got64, expected)

        # 2. Two 32-wide subgroups (e.g. NVIDIA)
        got32 = run_test(spv, 64, 32)
        self.assertEqual(got32, expected)

    def test_wave64_cross_half_isolation(self):
        """Lane in upper half cannot reach lower half even if address offset wraps."""
        # Lane 32 attempts to read target 0 of its half: must get lane 32 data, NOT lane 0 data
        addr_insns = ["v_mov_b32_e32 v3, 0"]
        data = [10 * i + 1 for i in range(64)]
        addr = [0] * 64
        expected = reference_bpermute(data, addr, ALL64, wave64=True)

        p = make_program(addr_insns, ALL64, wave64=True)
        spv = translate_text("cs_bperm_w64_iso", p, 64, wave64=True)

        got = run_test(spv, 64, 64)
        self.assertEqual(got, expected)
        # Low half gets data[0], high half gets data[32]
        self.assertTrue(all(v == data[0] for v in got[:32]))
        self.assertTrue(all(v == data[32] for v in got[32:]))

    def test_wave64_asymmetric_exec_masks(self):
        """Asymmetric EXEC mask across the two 64-lane halves."""
        # Low half active (0xFFFFFFFF), high half inactive (0x00000000)
        # Reversed address within half
        addr_insns = [
            "v_and_b32_e32 v4, 31, v0",
            "v_sub_nc_u32_e32 v4, 31, v4",
            "v_lshlrev_b32_e32 v3, 2, v4",
        ]
        exec_mask = 0x00000000FFFFFFFF  # low 32 on, high 32 off
        data = [10 * i + 1 for i in range(64)]
        addr = [4 * (31 - (i % 32)) for i in range(64)]
        expected = reference_bpermute(data, addr, exec_mask, wave64=True)

        p = make_program(addr_insns, exec_mask, wave64=True)
        spv = translate_text("cs_bperm_w64_asymm", p, 64, wave64=True)

        got = run_test(spv, 64, 64)
        self.assertEqual(got, expected)
        # Low half gets reversed data; high half preserves PRIOR_VD
        for i in range(32):
            self.assertEqual(got[i], data[31 - i])
        for i in range(32, 64):
            self.assertEqual(got[i], PRIOR_VD)

    def test_wave64_reverse(self):
        """Wave64 where each half reverses independently (lane l reads 31 - (l % 32) within its half)."""
        addr_insns = [
            "v_and_b32_e32 v4, 31, v0",
            "v_sub_nc_u32_e32 v4, 31, v4",
            "v_lshlrev_b32_e32 v3, 2, v4",
        ]
        data = [10 * i + 1 for i in range(64)]
        addr = [4 * (31 - (i % 32)) for i in range(64)]
        expected = reference_bpermute(data, addr, ALL64, wave64=True)

        p = make_program(addr_insns, ALL64, wave64=True)
        spv = translate_text("cs_bperm_w64_rev", p, 64, wave64=True)

        got64 = run_test(spv, 64, 64)
        self.assertEqual(got64, expected)

        got32 = run_test(spv, 64, 32)
        self.assertEqual(got32, expected)

    def test_wave64_broadcast_lane15_per_half(self):
        """Wave64 where each half broadcasts its lane 15 (lower gets 15, upper gets 47)."""
        addr_insns = ["v_mov_b32_e32 v3, 60"]  # 4 * 15 = 60
        data = [10 * i + 1 for i in range(64)]
        addr = [60] * 64
        expected = reference_bpermute(data, addr, ALL64, wave64=True)

        p = make_program(addr_insns, ALL64, wave64=True)
        spv = translate_text("cs_bperm_w64_bcast15", p, 64, wave64=True)

        got64 = run_test(spv, 64, 64)
        self.assertEqual(got64, expected)
        self.assertTrue(all(v == data[15] for v in got64[:32]))
        self.assertTrue(all(v == data[47] for v in got64[32:]))

        got32 = run_test(spv, 64, 32)
        self.assertEqual(got32, expected)

    def test_wave64_inactive_source_lane_reads_zero(self):
        """Wave64 alternating EXEC: active lanes reading inactive source lanes receive 0 in both halves."""
        # EXEC = 0x5555555555555555: even lanes active, odd lanes inactive
        # Even lane l reads lane (l % 32) + 1 within its half (odd lane, hence inactive)
        addr_insns = [
            "v_and_b32_e32 v4, 31, v0",
            "v_add_nc_u32_e32 v4, 1, v4",
            "v_and_b32_e32 v4, 31, v4",
            "v_lshlrev_b32_e32 v3, 2, v4",
        ]
        exec_mask = 0x5555555555555555
        data = [10 * i + 1 for i in range(64)]
        addr = [4 * (((i % 32) + 1) % 32) for i in range(64)]
        expected = reference_bpermute(data, addr, exec_mask, wave64=True)

        p = make_program(addr_insns, exec_mask, wave64=True)
        spv = translate_text("cs_bperm_w64_inact_src", p, 64, wave64=True)

        got64 = run_test(spv, 64, 64)
        self.assertEqual(got64, expected)

        got32 = run_test(spv, 64, 32)
        self.assertEqual(got32, expected)

    def test_wave64_single_active_lane(self):
        """Only lane 40 in upper half is active."""
        addr_insns = [
            "v_and_b32_e32 v4, 31, v0",
            "v_lshlrev_b32_e32 v3, 2, v4",
        ]
        exec_mask = 1 << 40
        data = [10 * i + 1 for i in range(64)]
        addr = [4 * (i % 32) for i in range(64)]
        expected = reference_bpermute(data, addr, exec_mask, wave64=True)

        p = make_program(addr_insns, exec_mask, wave64=True)
        spv = translate_text("cs_bperm_w64_single", p, 64, wave64=True)

        got64 = run_test(spv, 64, 64)
        self.assertEqual(got64, expected)
        self.assertEqual(got64[40], data[40])
        for i in range(64):
            if i != 40:
                self.assertEqual(got64[i], PRIOR_VD)

    def test_wave64_offset_wrapping(self):
        """Large offset:124 (31 words) with wrap-around in each 32-lane half."""
        addr_insns = ["v_lshlrev_b32_e32 v3, 2, v0"]
        offset = 124  # 31 words
        data = [10 * i + 1 for i in range(64)]
        addr = [4 * i for i in range(64)]
        expected = reference_bpermute(data, addr, ALL64, offset=offset, wave64=True)

        p = make_program(addr_insns, ALL64, offset=offset, wave64=True)
        spv = translate_text("cs_bperm_w64_off124", p, 64, wave64=True)

        got64 = run_test(spv, 64, 64)
        self.assertEqual(got64, expected)

        got32 = run_test(spv, 64, 32)
        self.assertEqual(got32, expected)


if __name__ == "__main__":
    unittest.main()
