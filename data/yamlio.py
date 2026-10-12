"""Shared strict YAML parsing: safe_load semantics, no aliases, no duplicate keys."""
from __future__ import annotations

from typing import Any

import yaml


class YamlError(ValueError):
    """Raised for malformed or disallowed YAML."""


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise YamlError(f"duplicate YAML key: {key!r}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def parse_strict_yaml(text: str) -> Any:
    try:
        # Aliases can expand a tiny file into a huge structure; not needed here.
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.AliasEvent):
                raise YamlError("YAML aliases/anchors are not allowed")
        return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 (SafeLoader subclass)
    except yaml.YAMLError as e:
        raise YamlError(f"invalid YAML: {e}") from e
