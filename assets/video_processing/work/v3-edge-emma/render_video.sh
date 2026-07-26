#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
video_root="$(cd -- "$script_dir/../.." && pwd)"

ffmpeg \
  -hide_banner \
  -y \
  -i "$video_root/source/codexforge_silent_master.mp4" \
  -i "$script_dir/codexforge_voiceover_v3.wav" \
  -filter_complex \
    "[0:v]ass=$script_dir/codexforge_voiceover_v3.ass[vout];[1:a]apad=pad_dur=1[aout]" \
  -map "[vout]" \
  -map "[aout]" \
  -t 179.68 \
  -c:v libx264 \
  -preset slow \
  -crf 18 \
  -pix_fmt yuv420p \
  -r 25 \
  -c:a aac \
  -b:a 192k \
  -ar 48000 \
  -ac 2 \
  -movflags +faststart \
  -metadata title="CodexForge: Built by Codex for Codex" \
  -metadata comment="OpenAI Build Week demo v3 with Emma neural voiceover and burned subtitles" \
  "$video_root/outputs/codexforge_demo_voiceover_v3.mp4"
