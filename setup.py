# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup


if __name__ == "__main__":
    # Package metadata and dependencies live in pyproject.toml. Keeping this
    # shim preserves compatibility with existing `pip install .` workflows.
    setup()
