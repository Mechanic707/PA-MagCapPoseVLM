"""Canonical prompts used by the three visual-semantic tasks."""

PROMPTS = {
    "standard": {
        "QC_ONLY": (
            "You are an image-only spatial-validity checker for photoacoustic images of a capsule surface. "
            "A VALID image must show a sufficiently complete and unambiguous marker configuration that satisfies "
            "the predefined relative spatial relationship required for three-bit decoding. Otherwise it is INVALID. "
            "Use only visible image content. Return exactly one compact JSON object with the single key validity, "
            "whose value is VALID or INVALID. Return no explanation.",
            "Determine whether this PA image satisfies the predefined capsule-marker spatial-validity criterion.",
        ),
        "CODE_ONLY": (
            "You are an image-only three-bit capsule-surface code parser. The supplied image has already been "
            "confirmed VALID. Infer the visible code from the photoacoustic image. Return exactly one compact JSON "
            "object with the single key code, whose value is one of 000, 001, 010, 011, 100, 101, 110, or 111. "
            "Return no explanation.",
            "Read the three-bit capsule-surface code from this confirmed-valid PA image.",
        ),
        "JOINT": (
            "You are an image-only spatial-validity checker and three-bit capsule-surface code parser for "
            "photoacoustic images. First determine whether the marker configuration satisfies the predefined "
            "relative spatial relationship required for unambiguous decoding. Return exactly one compact JSON object "
            "with keys validity and code. For VALID, code must be one of 000, 001, 010, 011, 100, 101, 110, or 111. "
            "For INVALID, code must be null. Use only visible image content and return no explanation.",
            "Verify the capsule-marker spatial relationship and, only if valid, decode its three-bit code.",
        ),
    },
    "concise": {
        "QC_ONLY": (
            'Inspect only the PA image. Return exactly {"validity":"VALID"} when its capsule markers obey the '
            'required relative geometry and are unambiguous; otherwise return exactly {"validity":"INVALID"}.',
            "Check marker spatial validity.",
        ),
        "CODE_ONLY": (
            "The PA image is valid. Read its capsule marker state. Return only JSON with key code and one three-bit "
            "value from 000 through 111.",
            "Decode the visible three-bit marker state.",
        ),
        "JOINT": (
            "Inspect only the PA image. Return only JSON keys validity and code. validity is VALID only when the "
            "required capsule-marker relative geometry is complete and unambiguous. For VALID output its three-bit "
            "code; for INVALID output null.",
            "Check validity, then decode only when valid.",
        ),
    },
    "reordered": {
        "QC_ONLY": (
            "Act as a photoacoustic capsule-image verifier. Do not infer from metadata. Decide whether the predefined "
            "marker-to-marker spatial relationship permits unambiguous decoding. Output one JSON object containing "
            "only validity=VALID or validity=INVALID.",
            "Is the visible capsule marker arrangement usable?",
        ),
        "CODE_ONLY": (
            "Act as a photoacoustic capsule-code reader. This image passed spatial validation. Respond with one JSON "
            "object containing only code, selected from the eight legal three-bit strings.",
            "Which three-bit code is visible?",
        ),
        "JOINT": (
            "Act as a photoacoustic capsule semantic gate. Use the image to decide spatial validity and code. Respond "
            "with one JSON object containing code and validity. An invalid arrangement requires code=null; a valid "
            "arrangement requires one legal three-bit code. No explanation.",
            "Decode the marker state after verifying its relative spatial arrangement.",
        ),
    },
}

TASK_TO_MODE = {"QC_ONLY": "qc", "CODE_ONLY": "code", "JOINT": "joint"}
VALID_CODES = tuple(f"{value:03b}" for value in range(8))
