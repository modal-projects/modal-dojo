import ast
import json
import tempfile
import unittest
from pathlib import Path

from modal_dojo.train_recipes.miles_recipe.muse_spark_image import (
    patch_muse_hf_cache_shards,
)

FIXTURE = Path(__file__).parent / "testdata/miles/muse_spark_checkpoint.py"


class MuseHFCacheShardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        target = self.base / "checkpoint.py"
        target.write_text(FIXTURE.with_suffix(".py.input").read_text())
        patch_muse_hf_cache_shards(target)
        patched = target.read_text()
        self.assertEqual(patched, FIXTURE.with_suffix(".py.output").read_text())
        patch_muse_hf_cache_shards(target)
        self.assertEqual(target.read_text(), patched)
        tree = ast.parse(patched)
        nodes = [
            node
            for node in tree.body
            if getattr(node, "name", "")
            in {"_dojo_resolve_safetensors_shard", "SafetensorsCheckpoint"}
        ]
        namespace = {
            "Path": Path,
            "json": json,
            "_read_shard_metadata": lambda path: {"weight": "spec"},
        }
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(target), "exec"),
            namespace,
        )
        self.namespace = namespace
        self.root = self.base / "models--org--model" / "snapshots" / ("a" * 40)
        self.root.mkdir(parents=True)
        (self.root / "config.json").write_text("{}")
        self.blobs = self.root.parent.parent / "blobs"
        self.blobs.mkdir()
        self.blob = self.blobs / ("b" * 64)
        self.blob.write_bytes(b"fixture")

    def load(self, filename):
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": filename}})
        )
        return self.namespace["SafetensorsCheckpoint"](self.root)

    def test_local(self):
        (self.root / "local.safetensors").write_bytes(b"fixture")
        self.assertEqual(
            self.load("local.safetensors")._shard_paths["local.safetensors"],
            self.root / "local.safetensors",
        )

    def test_canonical_cache_and_reader_uses_validated_target(self):
        link = self.root / "model.safetensors"
        link.symlink_to("../../blobs/" + self.blob.name)
        checkpoint = self.load(link.name)
        self.assertEqual(checkpoint._shard_paths[link.name], self.blob)
        # Changing the snapshot symlink after metadata inspection must not change
        # which validated file read_slice reads.
        outside = self.base / "outside"
        outside.write_bytes(b"unrelated")
        link.unlink()
        link.symlink_to(outside)
        opened = []

        class Reader:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def get_slice(self, name):
                return [123]

        def safe_open(path, **kwargs):
            opened.append(path)
            return Reader()

        self.namespace["safe_open"] = safe_open
        self.assertEqual(checkpoint.read_slice("weight", 0), 123)
        self.assertEqual(opened, [self.blob])

    def test_traversal_and_absolute(self):
        for name in ("../../blobs/" + self.blob.name, str(self.blob)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load(name)

    def test_arbitrary_symlink_escape(self):
        outside = self.base / "outside"
        outside.write_bytes(b"fixture")
        (self.root / "escape").symlink_to(outside)
        with self.assertRaises(ValueError):
            self.load("escape")

    def test_missing_and_broken(self):
        (self.root / "broken").symlink_to(self.blobs / "missing")
        for name in ("missing", "broken"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load(name)

    def test_sibling_repo_escape(self):
        other = self.base / "models--other--model" / "blobs" / ("c" * 64)
        other.parent.mkdir(parents=True)
        other.write_bytes(b"fixture")
        (self.root / "other").symlink_to(other)
        with self.assertRaises(ValueError):
            self.load("other")

    def test_blobs_symlink_escape(self):
        redirected = self.base / "redirected"
        self.blobs.rename(redirected)
        self.blobs.symlink_to(redirected)
        (self.root / "model.safetensors").symlink_to("../../blobs/" + self.blob.name)
        with self.assertRaises(ValueError):
            self.load("model.safetensors")

    def deduplicate_blob(self):
        global_blob = self.base / "blobs" / "0a" / ("0a" + "1" * 62)
        global_blob.parent.mkdir(parents=True)
        self.blob.rename(global_blob)
        self.blob.symlink_to("../../blobs/0a/" + global_blob.name)
        return global_blob

    def test_two_hop_deduplicated_cache(self):
        global_blob = self.deduplicate_blob()
        (self.root / "model.safetensors").symlink_to("../../blobs/" + self.blob.name)
        self.assertEqual(
            self.load("model.safetensors")._shard_paths["model.safetensors"],
            global_blob,
        )

    def test_direct_global_store_link_rejected(self):
        global_blob = self.deduplicate_blob()
        (self.root / "direct").symlink_to(global_blob)
        with self.assertRaises(ValueError):
            self.load("direct")

    def test_other_repo_deduplicated_link_rejected(self):
        global_blob = self.deduplicate_blob()
        other = self.base / "models--other--model" / "blobs" / ("c" * 64)
        other.parent.mkdir(parents=True)
        other.symlink_to(global_blob)
        (self.root / "other").symlink_to(other)
        with self.assertRaises(ValueError):
            self.load("other")

    def test_second_hop_external_path_rejected(self):
        external = self.base / "outside" / "0a" / ("0a" + "1" * 62)
        external.parent.mkdir(parents=True)
        self.blob.rename(external)
        self.blob.symlink_to(external)
        (self.root / "model.safetensors").symlink_to("../../blobs/" + self.blob.name)
        with self.assertRaises(ValueError):
            self.load("model.safetensors")

    def test_global_store_directory_symlink_rejected(self):
        self.deduplicate_blob()
        global_dir = self.base / "blobs"
        outside = self.base / "outside"
        global_dir.rename(outside)
        global_dir.symlink_to(outside)
        (self.root / "model.safetensors").symlink_to("../../blobs/" + self.blob.name)
        with self.assertRaises(ValueError):
            self.load("model.safetensors")

    def test_noncanonical_global_digest_path_rejected(self):
        global_blob = self.deduplicate_blob()
        noncanonical = global_blob.parent / "not-a-digest"
        global_blob.rename(noncanonical)
        self.blob.unlink()
        self.blob.symlink_to(noncanonical)
        (self.root / "model.safetensors").symlink_to("../../blobs/" + self.blob.name)
        with self.assertRaises(ValueError):
            self.load("model.safetensors")


if __name__ == "__main__":
    unittest.main()
