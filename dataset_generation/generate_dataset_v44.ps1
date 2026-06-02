$blender    = "C:\blender-2.78c-windows64\blender.exe"
$script     = "$PSScriptRoot\..\clevr-dataset-gen\image_generation\render_images.py"
$base_out   = "$PSScriptRoot\..\data\experiments\v44"
$clevr_data = "$PSScriptRoot\..\clevr-dataset-gen\image_generation\data"

$shape = "SmoothCube_v2"
$radius = 9
$elevation = 45
$azimuth = 90

$coords = @(
    @(0, -1),
    @(0, -0.5),
    @(0, 0),
    @(0, 0.5),
    @(0, 1),
    @(-1, 0),
    @(-0.5, 0),
    @(0.5, 0),
    @(1, 0)
)

$idx = 0

New-Item -ItemType Directory -Force -Path $base_out | Out-Null
foreach ($pair in $coords) {
    $x = $pair[0]
    $y = $pair[1]

    $xstr = $x.ToString() -replace '\\.', 'p' -replace '-', 'm'
    $ystr = $y.ToString() -replace '\\.', 'p' -replace '-', 'm'

    $out_img   = "$base_out\images\$shape"
    $out_scene = "$base_out\scenes\$shape"
    New-Item -ItemType Directory -Force -Path $out_img   | Out-Null
    New-Item -ItemType Directory -Force -Path $out_scene | Out-Null

    Write-Host "Rendering cube at fixed (x=$x, y=$y) radius=$radius, elevation=$elevation, azimuth=$azimuth (index=$idx)"

    & $blender --background -noaudio --python $script -- `
        --num_images 1 `
        --render_num_samples 64 `
        --shape $shape `
        --start_idx $idx `
        --output_image_dir $out_img `
        --output_scene_dir $out_scene `
        --filename_prefix "$shape`_r${radius}_a${azimuth}_x${xstr}_y${ystr}_n${idx}" `
        --width 512 `
        --height 512 `
        --key_light_jitter 0 `
        --fill_light_jitter 0 `
        --back_light_jitter 0 `
        --camera_jitter 0 `
        --object_xy_range 1 `
        --light_azimuth $azimuth `
        --light_elevation $elevation `
        --light_radius $radius `
        --world_bg_strength 0.2 `
        --fixed_size_name large `
        --fixed_object_x $x `
        --fixed_object_y $y `
        --base_scene_blendfile "$clevr_data\base_scene.blend" `
        --properties_json "$clevr_data\properties.json" `
        --shape_dir "$clevr_data\shapes" `
        --material_dir "$clevr_data\materials" `
        --output_scene_file "$out_scene\CLEVR_scenes.json"

    $idx++
}

Write-Host "Done! Outputs written to $base_out"
