# CodexForge video processing worktree

This directory contains the source, intermediate files, and rendered outputs for
the three-minute OpenAI Build Week demo.

The final v3 render is published as the
[CodexForge dashboard demo](https://youtu.be/R024p1wH2QI).

Large video, audio, and generated work-directory image files are intentionally
ignored by Git and remain local. The repository tracks the reproducible render
scripts, narration/subtitle metadata, architecture sources, and curated frames
and graphics embedded by the main README.

## Layout

- `source/`: the current silent master.
- `outputs/`: completed voiceover renders, including the retained v1 and v2
  versions and the current v3 render.
- `subtitles/`: editable SRT subtitle tracks for each render.
- `frames/`: frames exported from the edited master.
- `graphics/`: Mermaid source, rendered architecture graphics, and the
  historical legacy-mode terminal captures.
- `work/v1-piper-analysis/`: the first Piper narration and scene-analysis
  workspace, retained for provenance.
- `work/v2-piper-ryan/`: the complete local Piper v2 processing workspace,
  retained for provenance.
- `work/v3-edge-emma/`: the current Microsoft Emma neural-voice workspace,
  including the timing script, speech source, pronunciation text, intermediate
  audio, ASS captions, audition clips, and validation frames.

## Current voice and pronunciation

The v3 narration uses `en-US-EmmaMultilingualNeural` through `edge-tts`.
Subtitle text retains the exact technical spelling, while the synthesizer input
uses pronunciation-oriented text:

- `ONNX` is spoken as “onyx.”
- `JSONL` is spoken as “Jason Lines.”
- `ARC-GEN` is spoken as “ark gen.”
- `README` is spoken as “read me.”

## Install Python dependencies

Install the narration dependencies into the repository-local environment:

```bash
./.venv/bin/python -m pip install \
  -r assets/video_processing/requirements.txt
```

Piper is retained for the historical v1/v2 workspaces; the current v3 workflow
uses `edge-tts`. FFmpeg is also required and must be installed separately as a
system executable.

## Rebuilding v3

The narration step requires `edge-tts`; both steps require FFmpeg:

```bash
./.venv/bin/python \
  assets/video_processing/work/v3-edge-emma/render_voiceover.py

assets/video_processing/work/v3-edge-emma/render_video.sh
```

The first command generates the timed narration, ASS captions, and editable SRT.
The second burns the captions into the silent master and muxes the narration
into `outputs/codexforge_demo_voiceover_v3.mp4`.
