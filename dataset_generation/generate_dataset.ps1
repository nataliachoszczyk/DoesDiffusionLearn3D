$blender   = "C:\blender-2.78c-windows64\blender.exe"
$script    = "render_images.py"
$out_img   = "C:\Users\natal\Desktop\STUDIA MGR\SEM1\ZZSN\clevr-dataset-gen\output\images\v5"
$out_scene = "C:\Users\natal\Desktop\STUDIA MGR\SEM1\ZZSN\clevr-dataset-gen\output\scenes\v5"

$total_images = 800
$start_idx    = 0

for ($i = $start_idx; $i -lt ($start_idx + $total_images); $i++) {
  $az = [math]::Round((Get-Random -Minimum 0.0 -Maximum 360.0), 1)
  $el = [math]::Round((Get-Random -Minimum 30.0 -Maximum 75.0), 1)

  Write-Host "[$i] az=$az el=$el"

  & $blender --background -noaudio --python $script -- `
    --num_images 1 `
    --render_num_samples 128 `
    --light_azimuth $az `
    --light_elevation $el `
    --start_idx $i `
    --output_image_dir $out_img `
    --output_scene_dir $out_scene `
    --filename_prefix "LIGHT_v5" `
    --width 512 `
    --height 512 `
    --key_light_jitter 0 `
    --fill_light_jitter 0 `
    --back_light_jitter 0
}

Write-Host "Done! Generated $total_images images (idx $start_idx - $($start_idx + $total_images - 1))"