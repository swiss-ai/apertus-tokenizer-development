import unittest

from tokenization_scripts.runner.transforms import (
    MINHASH_UPSAMPLING,
    build_transforms,
    parse_transform_request,
)


class TransformRegistryTest(unittest.TestCase):
    def test_parses_a_versioned_named_transform(self):
        request = parse_transform_request(
            f'{MINHASH_UPSAMPLING}={{"column":"cluster_size"}}'
        )

        self.assertEqual(request.type, MINHASH_UPSAMPLING)
        self.assertEqual(request.parameters, {"column": "cluster_size"})
        self.assertEqual(request.descriptor()["version"], 1)

    def test_rejects_dataset_rendering_inside_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "unknown tokenization transform"):
            parse_transform_request("code_alchemy.placeholder_substitution")

    def test_builds_the_data_pipeline_transform(self):
        request = parse_transform_request(
            f'{MINHASH_UPSAMPLING}={{"weights":{{"1":2,"10":1}}}}'
        )

        transform = build_transforms([request])[0]

        self.assertEqual(transform.weights, {1: 2, 10: 1})
        self.assertEqual(transform.spec.type, MINHASH_UPSAMPLING)


if __name__ == "__main__":
    unittest.main()
