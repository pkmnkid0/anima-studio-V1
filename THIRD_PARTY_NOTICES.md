# Third-party code bundled with Anima Studio

`training_backend/` is vendored (bundled directly into this project, not
a separate download) from:

- **67372a/LoRA_Easy_Training_scripts_Backend** (the `backend/` folder of
  67372a/LoRA_Easy_Training_Scripts) - GNU General Public License v3.0.
  See `training_backend/LICENSE`.
- **67372a/sd-scripts** (vendored at `training_backend/sd_scripts/`, the
  actual training engine - a fork of kohya-ss/sd-scripts) - Apache
  License 2.0. See `training_backend/sd_scripts/LICENSE.md`.

This is what the Train tab actually runs when you click "Start
training" - Anima Studio provides the UI, config generation, and
process management around it, but does not reimplement the training
engine itself. Both projects' licenses are included unmodified in their
respective folders; nothing in this vendored code has been altered from
upstream.

Upstream projects:
- https://github.com/67372a/LoRA_Easy_Training_Scripts
- https://github.com/67372a/LoRA_Easy_Training_scripts_Backend
- https://github.com/67372a/sd-scripts


# MOSS-Image trainer

`moss_backend/sd_scripts/` is vendored from **RichyRich515/sd-scripts** (branch `moss-image`, a fork of
kohya-ss/sd-scripts) - Apache License 2.0, see `moss_backend/sd_scripts/LICENSE.md`. The Train tab's
"MOSS-Image" model type is based on **RichyRich515/LoRA_Easy_Training_Scripts** (branch `moss-image`).
It runs in its own virtual environment (`moss_backend/venv`), created from the Backend tab.

Changes made by Anima Studio to the vendored fork are listed in `moss_backend/PATCHES.md`.
Upstream: https://github.com/RichyRich515/sd-scripts


# KohakuMoss trainer

`kohaku_backend/` is vendored, unmodified, from **KohakuBlueLeaf/KohakuMossTrainer** (v0.1.0) under the
**KohakuMoss License 1.0** (Apache License 2.0 plus a naming requirement), see `kohaku_backend/LICENSE`. The Train
tab's "KohakuMoss" engine under the MOSS-Image model type runs it in its own virtual environment
(`kohaku_backend/venv`), created from the Backend tab. Anima Studio only writes its config files and starts its
`kogine` runner; none of its code is changed.

Section 10 of that licence (the "Naming Requirement") says a software product distributed to end users that
incorporates or is built upon the trainer must include "Kohaku" in its product name. It does not apply to internal
tools that are not distributed, nor to the LoRAs and images produced by running it. So: using Anima Studio on your own
machine is unaffected, but **if you distribute Anima Studio to other people with `kohaku_backend/` inside it, read
Section 10 first** - either leave `kohaku_backend/` out of what you share (the KohakuMoss engine then just reports
that the folder is missing) or comply with the naming term. This is not legal advice.
Upstream: https://github.com/KohakuBlueLeaf/KohakuMossTrainer


# AI edit tools (Edit tab)

The Magic eraser's AI tools use **IOPaint** (Sanster/IOPaint, formerly lama-cleaner) - Apache License 2.0 - which Anima
Studio does not bundle: it is installed with pip into its own environment (`edit_backend/venv`) from the Backend tab.
`edit_backend/` only contains Anima Studio's own catalogue and worker. IOPaint downloads model weights on first use
(LaMa, ZITS, MAT, FcF, LDM, Segment Anything, Real-ESRGAN, GFPGAN, RestoreFormer, rembg models, Stable Diffusion models,
...). Each model has its own licence, and some are restricted (for example non-commercial use) - check them before
commercial use.
Upstream: https://github.com/Sanster/IOPaint
