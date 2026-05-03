\"\"\"
DDPO package initialization.
Includes a global monkey patch for transformers compatibility.
\"\"\"

# --- Global Compatibility Patch for transformers ---
# Some packages (ImageReward) rely on internal transformers functions that 
# were moved in newer versions. We patch them here at the package root.
try:
    import transformers.modeling_utils
    import transformers.pytorch_utils
    moved_functions = [
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ]
    for name in moved_functions:
        if not hasattr(transformers.modeling_utils, name):
            if hasattr(transformers.pytorch_utils, name):
                setattr(transformers.modeling_utils, name, getattr(transformers.pytorch_utils, name))
except (ImportError, AttributeError):
    pass
# --------------------------------------------------
