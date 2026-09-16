import tempfile
import unittest
from pathlib import Path

from PIL import Image

from main import draw_preview, extract_fragments, prepare_image


FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")


class PrepareImageTests(unittest.TestCase):
    def test_applies_exif_orientation_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.jpg"
            output = root / "output"
            output.mkdir()

            exif = Image.Exif()
            exif[274] = 6

            Image.new("RGB", (120, 80), "white").save(
                source,
                exif=exif,
            )

            source_bytes = source.read_bytes()

            image, original = prepare_image(source, output)

            self.assertEqual(image.size, (80, 120))
            self.assertEqual(original.read_bytes(), source_bytes)
            self.assertTrue((output / "input.png").is_file())


class ExtractFragmentsTests(unittest.TestCase):
    def test_converts_ocr_result_to_fragments(self):
        polygon = [
            [10, 70],
            [60, 70],
            [60, 90],
            [10, 90],
        ]

        result = {
            "rec_texts": ["CMR"],
            "rec_scores": [0.9],
            "rec_polys": [polygon],
        }

        fragments = extract_fragments(result)

        self.assertEqual(fragments, [{
            "id": 1,
            "text": "CMR",
            "score": 0.9,
            "polygon": [
                [10.0, 70.0],
                [60.0, 70.0],
                [60.0, 90.0],
                [10.0, 90.0],
            ],
        }])

    def test_returns_empty_list_for_empty_result(self):
        result = {
            "rec_texts": [],
            "rec_scores": [],
            "rec_polys": [],
        }

        self.assertEqual(extract_fragments(result), [])

    def test_rejects_mismatched_result_lengths(self):
        result = {
            "rec_texts": ["CMR"],
            "rec_scores": [],
            "rec_polys": [],
        }

        with self.assertRaises(ValueError):
            extract_fragments(result)


class DrawPreviewTests(unittest.TestCase):
    @unittest.skipUnless(FONT_PATH.is_file(), "Arial font not found")
    def test_creates_preview_with_original_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            image = Image.new("RGB", (100, 120), "white")

            fragments = [{
                "id": 1,
                "text": "CMR",
                "score": 0.9,
                "polygon": [
                    [10.0, 70.0],
                    [60.0, 70.0],
                    [60.0, 90.0],
                    [10.0, 90.0],
                ],
            }]

            draw_preview(
                image,
                fragments,
                output,
                FONT_PATH,
            )

            preview_path = output / "preview.png"
            self.assertTrue(preview_path.is_file())

            with Image.open(preview_path) as preview:
                self.assertEqual(preview.size, image.size)


if __name__ == "__main__":
    unittest.main()