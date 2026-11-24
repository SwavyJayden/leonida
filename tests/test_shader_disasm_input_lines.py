"""shader_disasm.disasm() hands llvm-mc its input in lines of at most 64 bytes, never as one long line.

llvm-mc slows down sharply on one long input line: a 90 KB program took 58.6 s as a single line and 0.2 s in lines
of 64 bytes, with byte-identical disassembly (only the line/column numbers in its diagnostics change). The pipeline
library step disassembles 2,684 programs this way, so one line per program made it take hours.

subprocess.run is stubbed, so this needs no llvm-mc and no game data: it checks the text llvm-mc would receive.

    python -I tests/test_shader_disasm_input_lines.py
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ.setdefault("LLVM_MC", "llvm-mc")  # only builds the command line; nothing is started
import shader_disasm  # noqa: E402


class InputLinesTest(unittest.TestCase):
    def capture(self, code: bytes) -> str:
        """The `input` text disasm() passes to llvm-mc for `code`."""
        seen = {}

        def fake_run(cmd, input=None, **kwargs):
            seen["input"] = input
            return subprocess.CompletedProcess(cmd, 0, stdout="\ts_endpgm\n", stderr="")

        real = shader_disasm.subprocess.run
        shader_disasm.subprocess.run = fake_run
        try:
            asm, bad, err = shader_disasm.disasm(code, "gfx1013")
        finally:
            shader_disasm.subprocess.run = real
        self.assertEqual((asm, bad, err), ("\ts_endpgm\n", 0, ""))
        return seen["input"]

    @staticmethod
    def tokens(text: str) -> list:
        return text.split()

    def test_long_program_is_split_into_lines_of_64_bytes(self):
        code = bytes(range(256)) * 40  # 10,240 bytes
        text = self.capture(code)
        lines = text.split("\n")
        self.assertEqual(len(lines), 160)
        self.assertTrue(all(len(line.split(" ")) == 64 for line in lines), "every line holds 64 bytes")
        self.assertEqual(self.tokens(text), [f"0x{b:02x}" for b in code], "the same bytes, in order")

    def test_partial_last_line(self):
        code = bytes(range(100))
        lines = self.capture(code).split("\n")
        self.assertEqual([len(line.split(" ")) for line in lines], [64, 36])

    def test_short_input_stays_one_line(self):
        # The probe (s_endpgm) and any program of up to 64 bytes go on exactly as before.
        self.assertEqual(self.capture(shader_disasm.ENDPGM), shader_disasm.PROBE)
        code = bytes(range(64))
        self.assertEqual(self.capture(code), " ".join(f"0x{b:02x}" for b in code))

    def test_no_empty_or_trailing_lines(self):
        for size in (1, 63, 64, 65, 128, 129):
            with self.subTest(size=size):
                text = self.capture(bytes(size))
                self.assertFalse(text.endswith("\n"))
                self.assertNotIn("\n\n", text)
                self.assertEqual(len(self.tokens(text)), size)


if __name__ == "__main__":
    unittest.main()
