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
