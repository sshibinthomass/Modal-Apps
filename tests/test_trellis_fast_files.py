import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOST_FILE = ROOT / "llm-hosting" / "trellis-2-4b-fast.py"
INFERENCE_FILE = ROOT / "llm-inference" / "inf-trellis-2-4b-fast.py"


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def module_tree(path: Path) -> ast.Module:
    return ast.parse(read_file(path), filename=str(path))


def constant_value(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Constant {name} not found")


class TrellisFastFilesTest(unittest.TestCase):
    def test_host_uses_fast_modal_app_and_scale_to_zero(self):
        source = read_file(HOST_FILE)
        tree = module_tree(HOST_FILE)

        self.assertIn('modal.App("trellis-2-4b-fast")', source)
        self.assertEqual(constant_value(tree, "MIN_CONTAINERS"), 0)
        self.assertLessEqual(constant_value(tree, "SCALEDOWN_WINDOW"), 120)

    def test_host_defaults_prioritize_inference_speed(self):
        source = read_file(HOST_FILE)

        self.assertIn('pipeline_type: str = "512"', source)
        self.assertIn("decimation_target: int = 300_000", source)
        self.assertIn("texture_size: int = 1024", source)
        self.assertIn("remesh: bool = False", source)
        self.assertIn("webp: bool = False", source)
        self.assertIn("use_bf16: bool = False", source)
        self.assertIn('torch.autocast("cuda", dtype=torch.bfloat16)', source)
        self.assertIn("torch.inference_mode()", source)
        self.assertNotIn("verbose=True", source)

    def test_inference_client_targets_fast_app_with_fast_defaults(self):
        source = read_file(INFERENCE_FILE)
        tree = module_tree(INFERENCE_FILE)

        self.assertEqual(constant_value(tree, "APP_NAME"), "trellis-2-4b-fast")
        self.assertEqual(constant_value(tree, "CLASS_NAME"), "Trellis2FastModel")
        self.assertIn('default="512"', source)
        self.assertIn('"trellis-2-4b-fast-output.glb"', source)


if __name__ == "__main__":
    unittest.main()
