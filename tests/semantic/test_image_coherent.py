"""Coherent image accesses: glc on image_load / image_store (docs/design.md "Coherent image accesses").

On GFX10 a device-scope (coherent) load is glc + dlc and a device-scope store glc: Mesa ac_get_hw_cache_flags
(src/amd/common/ac_shader_util.c) sets exactly these bits for ACCESS_COHERENT. The game's POPS pixel shaders read
and write their ordered images that way, a read-modify-write that relies on seeing the previous fragment's store.
A read through the sampled array (SHADER-151's path for every other image_load) has no coherent form in SPIR-V, and
the fragment-shader interlock only makes Coherent memory visible between critical sections (Vulkan memory model:
non-atomic accesses are private unless Coherent). So a glc image_load reads the storage view the stores write
(OpImageRead), and a shader with coherent image accesses decorates its storage views Coherent.

These tests translate small pixel shaders with the real translator (tests/semantic/support.py) and check the SPIR-V:
which variable and component type every read and write uses, the decorations, capabilities and index features,
that plain accesses keep the sampled path, and that a coherent read the translator cannot model is refused.
"""
import os
import re
import shutil
import tempfile
import unittest

import support
import rdna2spirv

EPILOGUE = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
# dim -> (address registers, STORE_DIMS index)
DIMS = {"2D": ("v[4:5]", 0), "2D_ARRAY": ("v[4:6]", 1), "CUBE": ("v[4:6]", 1),
        "3D": ("v[4:6]", 2), "1D": ("v4", 3)}
CLASS_TYPES = {0: "%v4float", 1: "%v4uint_st", 2: "%v4int_st"}       # storage view class -> texel type


def translate(lines):
    return support.translate_lines("ps_coherent", ["v_mov_b32 v4, 1", "v_mov_b32 v5, 2", "v_mov_b32 v6, 0"]
                                   + lines + EPILOGUE)


def rmw(dim="2D", load_mods="glc dlc", store_mods="glc", dmask=0xF):
    """The POPS pattern: read a texel, add 1.0 to red, write it back to the same T#."""
    addr = DIMS[dim][0]
    return translate([f"image_load v[0:3], {addr}, s[0:7] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_{dim} unorm {load_mods}",
                      "s_waitcnt vmcnt(0)", "v_add_f32_e32 v0, 1.0, v0",
                      f"image_store v[0:3], {addr}, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_{dim} unorm {store_mods}"])


class Module:
    def __init__(self, text):
        self.text = text
        self.defs = {}
        self.ops = []
        for line in text.splitlines():
            m = re.match(r"\s*(%\w+) = (Op\w+)\s*(.*)$", line)
            if m:
                self.defs[m.group(1)] = (m.group(2), m.group(3).split())
                self.ops.append((m.group(1), m.group(2), m.group(3).split()))
            else:
                w = line.split()
                if w:
                    self.ops.append((None, w[0], w[1:]))
        self.caps = {a[0] for _, op, a in self.ops if op == "OpCapability"}
        self.coherent = {a[0] for _, op, a in self.ops if op == "OpDecorate" and a[1:] == ["Coherent"]}

    def variable(self, image):
        """The bindless array an image id was loaded from (OpLoad of an OpAccessChain into it)."""
        op, a = self.defs[image]
        assert op == "OpLoad", (image, op)
        op, a = self.defs[a[1]]
        assert op == "OpAccessChain", (image, op)
        return a[1]

    def reads(self):
        """(result type, variable) of every OpImageRead, in program order."""
        return [(a[0], self.variable(a[1])) for _, op, a in self.ops if op == "OpImageRead"]

    def writes(self):
        """(texel type, variable) of every OpImageWrite."""
        out = []
        for _, op, a in self.ops:
            if op == "OpImageWrite":
                texel_op, texel = self.defs[a[2]]
                assert texel_op == "OpCompositeConstruct", texel_op
                out.append((texel[0], self.variable(a[0])))
        return out

    def chains_into(self, prefix):
        return [a[1] for _, op, a in self.ops if op == "OpAccessChain" and a[1].startswith(prefix)]


class CoherentReadModifyWrite(unittest.TestCase):
    def test_coherent_load_reads_the_storage_views_the_store_writes(self):
        m = Module(rmw())
        reads, writes = m.reads(), m.writes()
        # One read per numeric class, each from that class's storage view, with that class's texel type.
        self.assertEqual(sorted(reads), sorted((CLASS_TYPES[c], f"%stv{c}_0") for c in range(3)))
        self.assertEqual(sorted(writes), sorted((CLASS_TYPES[c], f"%stv{c}_0") for c in range(3)))
        self.assertNotIn("OpImageFetch", m.text)                 # nothing reads through the sampled arrays
        self.assertEqual(m.chains_into("%images") + m.chains_into("%itx"), [])
        # Every read comes before every write (the read-modify-write order).
        order = [op for _, op, _ in m.ops if op in ("OpImageRead", "OpImageWrite")]
        self.assertEqual(order, ["OpImageRead"] * 3 + ["OpImageWrite"] * 3)

    def test_storage_views_are_coherent_and_the_read_feature_is_declared(self):
        text = rmw()
        m = Module(text)
        self.assertEqual(m.coherent, {"%stv0_0", "%stv1_0", "%stv2_0"})
        self.assertLessEqual({"StorageImageReadWithoutFormat", "StorageImageWriteWithoutFormat"}, m.caps)
        features = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, text)["features"]
        self.assertIn("shaderStorageImageReadWithoutFormat", features)

    def test_every_dimension_and_d16(self):
        for dim, (_, di) in DIMS.items():
            with self.subTest(dim=dim):
                m = Module(rmw(dim))
                self.assertEqual({v for _, v in m.reads()}, {f"%stv{c}_{di}" for c in range(3)})
                self.assertEqual(m.coherent, {f"%stv{c}_{di}" for c in range(3)})
        # d16: the same storage reads, the texel packed two halves per VGPR (as on the sampled path)
        m = Module(translate(["image_load v[0:1], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm glc dlc d16"]))
        self.assertEqual({v for _, v in m.reads()}, {"%stv0_0", "%stv1_0", "%stv2_0"})
        self.assertNotIn("OpImageFetch", m.text)

    def test_glc_alone_is_coherent(self):
        # GFX10: glc without dlc is shader-array scope for a load; Coherent (device scope) covers it.
        m = Module(rmw(load_mods="glc"))
        self.assertEqual(len(m.reads()), 3)
        self.assertEqual(m.coherent, {"%stv0_0", "%stv1_0", "%stv2_0"})


class OnlyCoherentAccessesChange(unittest.TestCase):
    def test_a_coherent_store_alone_needs_no_read_feature(self):
        m = Module(translate(["image_store v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm glc"]))
        self.assertEqual(m.coherent, {"%stv0_0", "%stv1_0", "%stv2_0"})
        self.assertNotIn("StorageImageReadWithoutFormat", m.caps)
        self.assertEqual(m.reads(), [])

    def test_plain_load_and_store_keep_the_sampled_path(self):
        # SHADER-151: a load without glc reads the sampled arrays, so it needs no storage read feature.
        m = Module(rmw(load_mods="", store_mods=""))
        self.assertEqual(m.reads(), [])
        self.assertIn("OpImageFetch", m.text)
        self.assertEqual(m.coherent, set())
        self.assertNotIn("StorageImageReadWithoutFormat", m.caps)

    def test_dlc_alone_is_not_coherent(self):
        # GFX10: dlc without glc is a CU-scope load that bypasses GL1 (ac_get_hw_cache_flags), not device scope.
        m = Module(rmw(load_mods="dlc", store_mods=""))
        self.assertEqual(m.reads(), [])
        self.assertEqual(m.coherent, set())


class CoherentMipLoad(unittest.TestCase):
    """image_load_mip glc (2D): the coherent read of mip 0 comes from the storage view; any other mip is zero."""

    def mip_load(self, mods="glc dlc", dmask=0xF, regs="v[0:3]"):
        return Module(translate([f"image_load_mip {regs}, v[4:6], s[0:7] dmask:0x{dmask:x} "
                                 f"dim:SQ_RSRC_IMG_2D unorm {mods}"]))

    def test_reads_the_coherent_storage_views_and_never_the_sampled_ones(self):
        m = self.mip_load()
        self.assertEqual(sorted(m.reads()), sorted((CLASS_TYPES[c], f"%stv{c}_0") for c in range(3)))
        self.assertEqual(m.coherent, {"%stv0_0", "%stv1_0", "%stv2_0"})
        self.assertNotIn("OpImageFetch", m.text)
        self.assertEqual(m.chains_into("%images") + m.chains_into("%itx"), [])
        self.assertLessEqual({"StorageImageReadWithoutFormat"}, m.caps)

    def test_the_read_is_taken_only_when_the_mip_is_zero(self):
        # v6 is the mip operand (v[4:6] = x, y, mip). The OpImageRead calls sit inside the branch taken on mip == 0.
        m = self.mip_load()
        zero = next(n for n, d in m.defs.items() if d == ("OpConstant", ["%uint", "0"]))
        branch = next(i for i, (_, op, a) in enumerate(m.ops)
                      if op == "OpBranchConditional" and m.defs.get(a[0], ("", []))[0] == "OpIEqual")
        self.assertEqual(m.defs[m.ops[branch][2][0]][1][-1], zero)
        merge = next(i for i, (r, op, _) in enumerate(m.ops) if op == "OpLabel" and r and r.endswith("_mipend"))
        reads = [i for i, (_, op, _) in enumerate(m.ops) if op == "OpImageRead"]
        self.assertEqual(len(reads), 3)
        self.assertTrue(all(branch < i < merge for i in reads))

    def test_other_mips_write_zero_to_every_result_register(self):
        for dmask, regs, n in ((0xF, "v[0:3]", 4), (0x3, "v[0:1]", 2), (0x5, "v[0:1]", 2)):
            with self.subTest(dmask=dmask):
                text = self.mip_load(dmask=dmask, regs=regs).text
                miss = text.split("_mipN = OpLabel")[1].split("_mipend = OpLabel")[0]
                self.assertEqual(len(re.findall(r"OpStore %reg_v", miss)), n, miss)

    def test_d16_packs_two_zero_halves_per_register(self):
        m = self.mip_load("glc dlc d16", dmask=0xF, regs="v[0:1]")
        self.assertEqual({v for _, v in m.reads()}, {"%stv0_0", "%stv1_0", "%stv2_0"})
        self.assertNotIn("OpImageFetch", m.text)

    def test_negative_control_plain_mip_load_stays_on_the_sampled_path(self):
        m = self.mip_load(mods="")
        self.assertEqual(m.reads(), [])
        self.assertIn("OpImageFetch", m.text)
        self.assertEqual(m.coherent, set())


class UnmodelledCoherentReads(unittest.TestCase):
    def check_refused(self, line):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate([line])

    def test_a_coherent_sampled_read_is_refused(self):
        self.check_refused("image_sample v[0:3], v[4:5], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D glc")
        self.check_refused("image_gather4_lz v[0:3], v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D glc")

    def test_a_coherent_load_without_a_storage_view_is_refused(self):
        self.check_refused("image_load v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm glc dlc")
        self.check_refused("image_load_mip v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY unorm glc dlc")
        self.check_refused("image_load_mip v[0:3], v[4:7], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D unorm glc dlc")
        self.check_refused("image_load_mip v[0:4], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm glc tfe")


@unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
class Validation(unittest.TestCase):
    def test_spirv_val_accepts_every_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            for dim in DIMS:
                with self.subTest(dim=dim):
                    sa, sv = os.path.join(tmp, f"{dim}.spvasm"), os.path.join(tmp, f"{dim}.spv")
                    with open(sa, "w", encoding="utf-8") as f:
                        f.write(rmw(dim))
                    result = rdna2spirv.assemble_spirv(sa, sv)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_spirv_val_accepts_a_coherent_mip_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, mods, regs in (("f32", "glc dlc", "v[0:3]"), ("d16", "glc dlc d16", "v[0:1]")):
                with self.subTest(form=name):
                    text = translate([f"image_load_mip {regs}, v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm {mods}"])
                    sa, sv = os.path.join(tmp, f"{name}.spvasm"), os.path.join(tmp, f"{name}.spv")
                    with open(sa, "w", encoding="utf-8") as f:
                        f.write(text)
                    result = rdna2spirv.assemble_spirv(sa, sv)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
