$blender    = "C:\blender-2.78c-windows64\blender.exe"
$script     = "$PSScriptRoot\..\clevr-dataset-gen\image_generation\render_images.py"
$base_out   = "$PSScriptRoot\..\data\experiments\v38"
$clevr_data = "$PSScriptRoot\..\clevr-dataset-gen\image_generation\data"

$shape = "SmoothCube_v2"
# 9 radius values (sensible spread from near to far)
$radii = @(2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.5, 10.0)

$idx = 0

New-Item -ItemType Directory -Force -Path $base_out | Out-Null
foreach ($r in $radii) {
    $out_img   = "$base_out\images\$shape"
    $out_scene = "$base_out\scenes\$shape"
    New-Item -ItemType Directory -Force -Path $out_img   | Out-Null
    New-Item -ItemType Directory -Force -Path $out_scene | Out-Null

    Write-Host "Rendering 1 image for shape=$shape with light_radius=$r (index=$idx)"

    & $blender --background -noaudio --python $script -- `
        --num_images 1 `
        --render_num_samples 64 `
        --shape $shape `
        --start_idx $idx `
        --output_image_dir $out_img `
        --output_scene_dir $out_scene `
        --filename_prefix "$shape`_r${r}" `
        --width 512 `
        --height 512 `
        --key_light_jitter 0 `
        --fill_light_jitter 0 `
        --back_light_jitter 0 `
        --camera_jitter 0 `
        --light_azimuth 90 `
        --light_elevation 45 `
        --light_radius $r `
        --base_scene_blendfile "$clevr_data\base_scene.blend" `
        --properties_json "$clevr_data\properties.json" `
        --shape_dir "$clevr_data\shapes" `
        --material_dir "$clevr_data\materials" `
        --output_scene_file "$out_scene\CLEVR_scenes.json"

    $idx++
}

Write-Host "Done! Outputs written to $base_out"