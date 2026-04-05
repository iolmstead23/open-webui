"""Operation registry -- typed catalog of every Obsidian REST operation via mcpo.

Dependencies: config (for mcpo base URL).
Exposed interface: OPERATIONS dict, get_operation() lookup.
"""
from typing import Any

# Risk classifications
READ = "READ"
WRITE = "WRITE"
DESTRUCTIVE = "DESTRUCTIVE"


def _build_operation_entry(
    name: str,
    endpoint: str,
    risk: str,
    description: str,
    required_fields: dict[str, str],
    optional_fields: dict[str, tuple[str, Any]] | None = None,
    notes: str = "",
) -> dict:
    """Build a registry entry."""
    return {
        "name": name,
        "endpoint": endpoint,
        "risk": risk,
        "description": description,
        "required_fields": required_fields,
        "optional_fields": optional_fields or {},
        "notes": notes,
    }


OPERATIONS: dict[str, dict] = {
    "obsidian_list_files_in_dir": _build_operation_entry(
        name="obsidian_list_files_in_dir",
        endpoint="/obsidian/obsidian_list_files_in_dir",
        risk=READ,
        description="List files in a specific vault directory.",
        required_fields={"dirpath": "str"},
        notes="Empty directories not returned. No trailing slash on dirpath.",
    ),
    "obsidian_list_files_in_vault": _build_operation_entry(
        name="obsidian_list_files_in_vault",
        endpoint="/obsidian/obsidian_list_files_in_vault",
        risk=READ,
        description="List all files in the vault root.",
        required_fields={},
    ),
    "obsidian_get_file_contents": _build_operation_entry(
        name="obsidian_get_file_contents",
        endpoint="/obsidian/obsidian_get_file_contents",
        risk=READ,
        description="Get content of a single file.",
        required_fields={"filepath": "str"},
    ),
    "obsidian_simple_search": _build_operation_entry(
        name="obsidian_simple_search",
        endpoint="/obsidian/obsidian_simple_search",
        risk=READ,
        description="Simple text search across all vault files.",
        required_fields={"query": "str"},
        optional_fields={"context_length": ("int", 100)},
        notes="Use when user asks for notes containing specific text.",
    ),
    "obsidian_complex_search": _build_operation_entry(
        name="obsidian_complex_search",
        endpoint="/obsidian/obsidian_complex_search",
        risk=READ,
        description="Complex search using JsonLogic query.",
        required_fields={"query": "dict"},
        notes="Use for structural queries (tags, folders, dates). Supports glob and regexp operators.",
    ),
    "obsidian_batch_get_file_contents": _build_operation_entry(
        name="obsidian_batch_get_file_contents",
        endpoint="/obsidian/obsidian_batch_get_file_contents",
        risk=READ,
        description="Get contents of multiple files concatenated with headers.",
        required_fields={"filepaths": "list[str]"},
        notes="Prefer over multiple obsidian_get_file_contents calls.",
    ),
    "obsidian_get_periodic_note": _build_operation_entry(
        name="obsidian_get_periodic_note",
        endpoint="/obsidian/obsidian_get_periodic_note",
        risk=READ,
        description="Get current periodic note for a period type.",
        required_fields={"period": "str"},
        notes="period must be: daily, weekly, monthly, quarterly, yearly.",
    ),
    "obsidian_get_recent_periodic_notes": _build_operation_entry(
        name="obsidian_get_recent_periodic_notes",
        endpoint="/obsidian/obsidian_get_recent_periodic_notes",
        risk=READ,
        description="Get recent periodic notes for a period type.",
        required_fields={"period": "str"},
        optional_fields={
            "limit": ("int", 5),
            "include_content": ("bool", False),
        },
    ),
    "obsidian_get_recent_changes": _build_operation_entry(
        name="obsidian_get_recent_changes",
        endpoint="/obsidian/obsidian_get_recent_changes",
        risk=READ,
        description="Get recently modified files.",
        required_fields={},
        optional_fields={
            "limit": ("int", 10),
            "days": ("int", 90),
        },
        notes="Requires Dataview plugin. Uses DQL query internally.",
    ),
    "obsidian_patch_content": _build_operation_entry(
        name="obsidian_patch_content",
        endpoint="/obsidian/obsidian_patch_content",
        risk=WRITE,
        description="Insert content relative to a heading, block, or frontmatter field.",
        required_fields={
            "filepath": "str",
            "operation": "str",
            "target_type": "str",
            "target": "str",
            "content": "str",
        },
        notes="Prefer over append when a location anchor is specified. prepend+frontmatter is invalid.",
    ),
    "obsidian_append_content": _build_operation_entry(
        name="obsidian_append_content",
        endpoint="/obsidian/obsidian_append_content",
        risk=WRITE,
        description="Append content to end of a file.",
        required_fields={"filepath": "str", "content": "str"},
        notes="Use only when no location anchor is specified.",
    ),
    "obsidian_delete_file": _build_operation_entry(
        name="obsidian_delete_file",
        endpoint="/obsidian/obsidian_delete_file",
        risk=DESTRUCTIVE,
        description="Delete a file or directory from the vault.",
        required_fields={"filepath": "str", "confirm": "bool"},
        notes="confirm must be true. Planner must not include unless user says delete/remove.",
    ),
    "obsidian_get_tags": _build_operation_entry(
        name="obsidian_get_tags",
        endpoint="/obsidian/obsidian_get_tags",
        risk=READ,
        description="Get all tags in the vault with counts.",
        required_fields={},
        notes="Returns inline (#tag) and frontmatter tags with hierarchical parent counts.",
    ),
}


def get_operation(name: str) -> dict | None:
    """Look up an operation by name. Returns None if not found."""
    return OPERATIONS.get(name)
