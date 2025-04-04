from base import *

import os
import unittest


class TestUtilFunc(unittest.TestCase):
    def test_mut_range(self):
        self.assertListEqual(which_muts, range(102, 261 + 1))

    def test_bps(self):
        self.assertListEqual(
            primerize_degen.util.diff_bps(".....(((((.....)))))....."),
            [[(6, 20), (7, 19), (8, 18), (9, 17), (10, 16)]],
        )
        self.assertListEqual(
            primerize_degen.util.diff_bps(".....((.((.....)).))....."),
            [[(9, 17), (10, 16)], [(6, 20), (7, 19)]],
        )
        self.assertListEqual(
            primerize_degen.util.diff_bps(
                [".....(((((.....))))).....", ".....((((......).)))....."], 10
            ),
            [[(-1, 7), (0, 6)], [(-1, 6)]],
        )
        self.assertListEqual(
            primerize_degen.util.diff_bps(
                [".....(((((.....))))).....", ".....((((......).)))....."], -10
            ),
            [[(19, 27), (20, 26)], [(19, 26)]],
        )
        self.assertListEqual(
            primerize_degen.util.diff_bps(".....[[(((...]]..)))....."),
            [[(6, 15), (7, 14)], [(8, 20), (9, 19), (10, 18)]],
        )
        self.assertListEqual(
            primerize_degen.util.diff_bps(
                [".....(((((.....))))).....", ".....(((((.....)))))....."]
            ),
            [],
        )
        self.assertListEqual(
            primerize_degen.util.diff_bps(
                [".....(((((.....))))).....", ".....(((((.....)))))....."], flag=False
            ),
            [[(6, 20), (7, 19), (8, 18), (9, 17), (10, 16)]],
        )
        self.assertListEqual(
            primerize_degen.util.diff_bps(
                [".....(((((.....))))).....", ".....((((.(...).))))....."], flag=False
            ),
            [[(6, 20), (7, 19), (8, 18), (9, 17), (10, 16)], [(11, 15)]],
        )

    def test_get_mut(self):
        for i, nt in enumerate("ACGT"):
            self.assertEqual(primerize_degen.util.get_mutation(nt, 1), "TGCA"[i])
        for i, nt in enumerate("ACGT"):
            self.assertEqual(primerize_degen.util.get_mutation(nt, 2), "CAAC"[i])
        for i, nt in enumerate("ACGT"):
            self.assertEqual(primerize_degen.util.get_mutation(nt, 3), "GTTG"[i])
        for i, nt in enumerate("ACGT"):
            self.assertEqual(primerize_degen.util.get_mutation(nt, 4), "CATG"[i])

    def test_valid_pair(self):
        self.assertTrue(primerize_degen.util.valid_WC_pair("A", "U"))
        self.assertTrue(primerize_degen.util.valid_WC_pair("A", "T"))
        self.assertTrue(primerize_degen.util.valid_WC_pair("U", "A"))
        self.assertTrue(primerize_degen.util.valid_WC_pair("T", "A"))
        self.assertTrue(primerize_degen.util.valid_WC_pair("C", "G"))
        self.assertTrue(primerize_degen.util.valid_WC_pair("G", "C"))
        self.assertTrue(primerize_degen.util.valid_WC_pair("G", "U"))
        self.assertTrue(primerize_degen.util.valid_WC_pair("G", "T"))
        self.assertTrue(primerize_degen.util.valid_WC_pair("U", "G"))
        self.assertTrue(primerize_degen.util.valid_WC_pair("T", "G"))
        self.assertFalse(primerize_degen.util.valid_WC_pair("U", "U"))
        self.assertFalse(primerize_degen.util.valid_WC_pair("A", "A"))
        self.assertFalse(primerize_degen.util.valid_WC_pair("C", "C"))
        self.assertFalse(primerize_degen.util.valid_WC_pair("G", "G"))
        self.assertFalse(primerize_degen.util.valid_WC_pair("C", "A"))
        self.assertFalse(primerize_degen.util.valid_WC_pair("C", "U"))

    def test_num2coord(self):
        self.assertEqual(primerize_degen.util.num_to_coord(1), "A01")
        self.assertEqual(primerize_degen.util.num_to_coord(96), "H12")
        self.assertEqual(primerize_degen.util.num_to_coord(9), "A02")
        self.assertIsNone(primerize_degen.util.num_to_coord(0))
        self.assertIsNone(primerize_degen.util.num_to_coord(97))
        self.assertIsNone(primerize_degen.util.num_to_coord(None))
        self.assertIsNone(primerize_degen.util.num_to_coord(11.5))
        self.assertIsNone(primerize_degen.util.num_to_coord("a"))

    def test_coord2num(self):
        self.assertEqual(primerize_degen.util.coord_to_num("A01"), 1)
        self.assertEqual(primerize_degen.util.coord_to_num("H12"), 96)
        self.assertEqual(primerize_degen.util.coord_to_num("A02"), 9)
        self.assertIsNone(primerize_degen.util.coord_to_num("A00"))
        self.assertIsNone(primerize_degen.util.coord_to_num("H13"))
        self.assertIsNone(primerize_degen.util.coord_to_num("X10"))
        self.assertIsNone(primerize_degen.util.coord_to_num("ABC"))
        self.assertIsNone(primerize_degen.util.coord_to_num(None))
        self.assertIsNone(primerize_degen.util.coord_to_num(1))

    def test_seq(self):
        self.assertEqual(primerize_degen.util.DNA2RNA("ACGTU"), "ACGUU")
        self.assertEqual(primerize_degen.util.RNA2DNA("ACGTU"), "ACGTT")
        self.assertEqual(primerize_degen.util.RNA2DNA("ABCDE"), "ABCDE")
        self.assertEqual(primerize_degen.util.reverse("ACGTU"), "UTGCA")
        self.assertEqual(primerize_degen.util.complement("ACGTU"), "TGCAA")
        self.assertEqual(primerize_degen.util.reverse_complement("ACGTU"), "AACGT")
        self.assertRaises(ValueError, primerize_degen.util.complement, "ABCDE")

    def test_suffix(self):
        self.assertIn("R", primerize_degen.util_func._primer_suffix(1))
        self.assertIn("F", primerize_degen.util_func._primer_suffix(2))


class TestUtilClass(unittest.TestCase):
    def test_mutation(self):
        inst = primerize_degen.Mutation()
        self.assertEqual(len(inst), 0)
        inst.push("G12C")
        inst.push(["A11U"])
        self.assertTrue("G12C" in inst)
        self.assertTrue(["A11T"] in inst)
        self.assertTrue(["A11U"] in inst)
        self.assertFalse("G12A" in inst)
        self.assertEqual(len(inst), 2)

        other = primerize_degen.Mutation(["G12C", "A11T"])
        self.assertEqual(inst, other)
        self.assertTrue(other.pop("G12C"))
        self.assertEqual(len(other), 1)
        self.assertNotEqual(inst, other)
        self.assertTrue(other.pop("A11U"))
        self.assertFalse(other.pop("A11T"))
        self.assertEqual(len(other), 0)
        self.assertFalse(inst.pop(["G12U"]))

        print(inst.echo())
        self.assertTrue(inst.pop(["G12C", "A11T"]))
        self.assertEqual(inst, other)
        self.assertRaises(ValueError, inst.push, "G12G")

        other.push("A100C")
        inst.merge(other)
        self.assertEqual(inst, other)
        other = primerize_degen.Mutation(["C99G", "G101T"])
        inst.merge(other)
        self.assertNotEqual(inst, other)

    def test_construct_list(self):
        inst = primerize_degen.Construct_List()
        self.assertEqual(len(inst), 1)
        inst.pop("WT")
        self.assertEqual(len(inst), 0)

        inst = primerize_degen.Construct_List()
        self.assertEqual(inst._data[0], "WT")
        inst.push(primerize_degen.Mutation("G12C"))
        self.assertFalse(inst.push("G12C"))
        self.assertTrue(inst.push("A11U"))
        self.assertTrue(inst.push(primerize_degen.Mutation(["C1A", "G12C"])))
        self.assertEqual(len(inst), 4)
        self.assertTrue("G12C" in inst)
        self.assertTrue(["G12C", "C1A"] in inst)
        self.assertTrue(primerize_degen.Mutation(["G12C", "C1A"]) in inst)
        self.assertFalse("C1A" in inst)

        self.assertFalse(inst.pop("C1A"))
        self.assertTrue(inst.pop("G12C"))
        self.assertEqual(len(inst), 3)
        self.assertTrue(inst.pop(primerize_degen.Mutation(["C1A", "G12C"])))
        print(inst.echo())

        other = primerize_degen.Construct_List()
        other.push("C11A")
        inst.merge(other)
        self.assertEqual(len(inst), 3)
        repeat = inst.merge(other)
        self.assertEqual(len(inst), 3)
        self.assertEqual(len(repeat), 1)

    def test_96well_plate(self):
        inst = primerize_degen.Plate_96Well()
        self.assertEqual(len(inst), 0)
        inst.set("A01", "WT", "ACCTTG")
        self.assertRaises(AttributeError, inst.set, "A00", "WT", "ACCTTG")
        inst.set("C02", "Lib1-G6T", "ACCTTT")
        inst.set("A01", "Lib1-A1G", "GCCTTG")
        self.assertEqual(len(inst), 2)

        self.assertRaises(AttributeError, inst.get, "X")
        self.assertRaises(AttributeError, inst.get, "B13")
        self.assertRaises(KeyError, inst.get, "A12")
        self.assertTupleEqual(inst.get("A01"), ("Lib1-A1G", "GCCTTG"))

        print(repr(inst))
        print(inst.echo())
        inst.save()
        os.remove("plate.svg")
        inst.reset()
        self.assertEqual(len(inst), 0)


if __name__ == "__main__":
    unittest.main()
