input="$1"
output="$2"

ffmpeg -framerate 10 -i ~/workspaces/lost-tools/outputs/$input/output_depth_estimator_%06d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ~/workspaces/lost-tools/outputs/$output/"$output"_depth_estimator.mp4 -y

ffmpeg -framerate 10 -i ~/workspaces/lost-tools/outputs/$input/output_dynamic_sg_%06d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ~/workspaces/lost-tools/outputs/$output/"$output"_dynamic_sg.mp4 -y

ffmpeg -framerate 10 -i ~/workspaces/lost-tools/outputs/$input/output_pose_estimator_%06d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ~/workspaces/lost-tools/outputs/$output/"$output"_pose_estimator.mp4 -y

ffmpeg -framerate 10 -i ~/workspaces/lost-tools/outputs/$input/output_sgg3d_%06d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ~/workspaces/lost-tools/outputs/$output/"$output"_sgg3d.mp4 -y

ffmpeg -framerate 10 -i ~/workspaces/lost-tools/outputs/$input/output_tracker_%06d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ~/workspaces/lost-tools/outputs/$output/"$output"_tracker.mp4 -y