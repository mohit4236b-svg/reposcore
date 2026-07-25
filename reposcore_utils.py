"""
Shared preprocessing used by BOTH train_model.py and app.py.

Keeping this in one place matters: if the training script and the Streamlit
app clean README text differently, you get train/serve skew — the model was
fit on one distribution of text and scores a different one at inference time.
Import strip_badges from here in both places instead of copy-pasting it.
"""

import re

# Strips markdown image/badge syntax, shields.io/badge-service URLs, and
# common CI/build/status badge hosts. This exists because has_ci/has_tests
# are excluded as direct model features (they were used to build the label),
# but their *signal* was leaking back in indirectly through badge markup
# embedded in the README text (tokens like "workflows", "shields", "badge",
# "svg" ranked in the top-15 TF-IDF features before this was applied).
BADGE_PATTERNS = [
    r"!\[[^\]]*\]\([^)]*\)",                 # ![alt](url) markdown images
    r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)",     # linked badge images
    r"https?://\S*(shields\.io|badge\.fury\.io|travis-ci|"
    r"github\.com/\S*workflows\S*|circleci|codecov|coveralls|"
    r"bestpractices\.coreinfrastructure|securityscorecards|"
    r"oss-fuzz|ossrank|zenodo)\S*",
]


def strip_badges(text: str) -> str:
    """Remove badge/shield markup from README text before vectorizing."""
    if not isinstance(text, str):
        return ""
    for pattern in BADGE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text
