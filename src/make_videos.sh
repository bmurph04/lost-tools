#!/usr/bin/env bash
#
# Turn a run's per-frame visualizations into one mp4 per module.
#
# Usage: src/make_videos.sh <input-run> <output-run> [framerate]
#   e.g. src/make_videos.sh output_streamed output_streamed 5
#
# Frame numbers come from the pipeline's frame clock `t`, so a module that only
# runs on some frames (anything behind sg_interval) or only starts partway
# through (dynamic_sg, which waits for warmup_frames) leaves gaps and does not
# start at zero. ffmpeg's %06d pattern cannot express that -- it stops at the
# first missing index -- so glob the files instead and let ffmpeg order them.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <input-run> <output-run> [framerate]" >&2
    exit 1
fi

input="$1"
output="$2"
framerate="${3:-5}"

# Resolve paths from the script's own location so this works from any cwd and
# on any machine, rather than assuming ~/workspace/lost-tools.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input_dir="$repo_root/outputs/$input"
output_dir="$repo_root/outputs/$output"

if [[ ! -d "$input_dir" ]]; then
    echo "error: no run at $input_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"

# Must match the output_suffix values in configs/lost_tools.yaml
modules=(
    output_detector
    output_tracker
    output_depth_provider
    output_pose_provider
    output_point_lifter
    output_sgg3d
    output_dynamic_sg
)

made=0
for module in "${modules[@]}"; do
    module_dir="$input_dir/$module"

    if [[ ! -d "$module_dir" ]]; then
        echo "[skip] $module: no directory"
        continue
    fi

    count=$(find "$module_dir" -maxdepth 1 -type f -name "${module}_*.jpg" | wc -l)
    if [[ "$count" -eq 0 ]]; then
        echo "[skip] $module: no frames"
        continue
    fi

    target="$output_dir/${output}_${module#output_}.mp4"
    echo "[make] $module: $count frames -> $(basename "$target")"

    # -pattern_type glob tolerates gaps and a non-zero starting index. The scale
    # filter rounds to even dimensions, which libx264/yuv420p requires.
    ffmpeg -hide_banner -loglevel error \
        -framerate "$framerate" \
        -pattern_type glob -i "$module_dir/${module}_*.jpg" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -c:v libx264 -pix_fmt yuv420p \
        "$target" -y

    made=$((made + 1))
done

echo "done: $made video(s) in $output_dir"