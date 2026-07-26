from __future__ import annotations

import json
from pathlib import Path
import subprocess
import textwrap
import wave

from piper import PiperVoice, SynthesisConfig


ROOT = Path(__file__).resolve().parents[4]
WORK = Path(__file__).resolve().parent
MODEL = Path("/home/rohit-raje/.cache/piper-voices/en_US-ryan-high.onnx")
SEGMENTS_PATH = WORK / "voiceover_segments.json"
NARRATION_WAV = WORK / "codexforge_voiceover.wav"
SRT_PATH = (
    ROOT
    / "assets"
    / "video_processing"
    / "subtitles"
    / "codexforge_voiceover.srt"
)
ASS_PATH = WORK / "codexforge_voiceover.ass"
VIDEO_DURATION = 180.565


def timestamp_srt(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def timestamp_ass(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cents = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cents:02d}"


def wrap_subtitle(text: str) -> str:
    return "\n".join(textwrap.wrap(text, width=68, break_long_words=False))


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


segments = json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
voice = PiperVoice.load(MODEL)
syn_config = SynthesisConfig(length_scale=1.04, volume=0.92)

processed: list[Path] = []
for index, segment in enumerate(segments, start=1):
    raw_path = WORK / f"voice-{index:02d}-raw.wav"
    final_path = WORK / f"voice-{index:02d}.wav"
    available = float(segment["end"]) - float(segment["start"]) - 0.12
    if not final_path.is_file():
        with wave.open(str(raw_path), "wb") as wav_file:
            voice.synthesize_wav(segment["spoken"], wav_file, syn_config=syn_config)

        duration = wav_duration(raw_path)
        speed = max(1.0, duration / available)
        if speed > 1.18:
            raise RuntimeError(
                f"segment {index} needs excessive speed-up: "
                f"{duration:.2f}s into {available:.2f}s ({speed:.3f}x)"
            )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(raw_path),
                "-af",
                (
                    f"atempo={speed:.6f},"
                    "aresample=48000,"
                    "highpass=f=70,"
                    "afade=t=in:st=0:d=0.035,"
                    f"afade=t=out:st={max(0.0, min(duration / speed, available) - 0.07):.4f}:d=0.07"
                ),
                "-ac",
                "1",
                str(final_path),
            ],
            check=True,
        )
    processed.append(final_path)
    print(
        f"{index:02d}: window={available:.2f}s final={wav_duration(final_path):.2f}s",
        flush=True,
    )

srt_blocks = []
ass_events = []
for index, segment in enumerate(segments, start=1):
    wrapped = wrap_subtitle(segment["subtitle"])
    srt_blocks.append(
        f"{index}\n"
        f"{timestamp_srt(float(segment['start']))} --> "
        f"{timestamp_srt(float(segment['end']))}\n"
        f"{wrapped}\n"
    )
    ass_text = wrapped.replace("\n", r"\N").replace("{", r"\{").replace("}", r"\}")
    ass_events.append(
        "Dialogue: 0,"
        f"{timestamp_ass(float(segment['start']))},"
        f"{timestamp_ass(float(segment['end']))},"
        f"Voiceover,,0,0,0,,{ass_text}"
    )

SRT_PATH.write_text("\n".join(srt_blocks), encoding="utf-8")
ASS_PATH.write_text(
    "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1920",
            "PlayResY: 1080",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            (
                "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
                "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,"
                "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
                "Alignment,MarginL,MarginR,MarginV,Encoding"
            ),
            (
                "Style: Voiceover,DejaVu Sans,34,&H00FFFFFF,&H000000FF,"
                "&H00101010,&H00000000,0,0,0,0,100,100,0,0,1,2.2,0,"
                "2,90,90,16,1"
            ),
            "",
            "[Events]",
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
            *ass_events,
            "",
        ]
    ),
    encoding="utf-8",
)

command = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-f",
    "lavfi",
    "-t",
    str(VIDEO_DURATION),
    "-i",
    "anullsrc=r=48000:cl=mono",
]
for path in processed:
    command.extend(["-i", str(path)])

filter_parts = []
mix_inputs = ["[0:a]"]
for index, (path, segment) in enumerate(zip(processed, segments), start=1):
    delay = round(float(segment["start"]) * 1000)
    label = f"v{index}"
    filter_parts.append(f"[{index}:a]adelay={delay}:all=1[{label}]")
    mix_inputs.append(f"[{label}]")
filter_parts.append(
    "".join(mix_inputs)
    + f"amix=inputs={len(mix_inputs)}:duration=first:normalize=0,"
    + "loudnorm=I=-16:TP=-1.5:LRA=7[aout]"
)
command.extend(
    [
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[aout]",
        "-t",
        str(VIDEO_DURATION),
        "-ac",
        "2",
        "-ar",
        "48000",
        str(NARRATION_WAV),
    ]
)
subprocess.run(command, check=True)
print(f"Wrote {NARRATION_WAV}")
print(f"Wrote {SRT_PATH}")
print(f"Wrote {ASS_PATH}")
