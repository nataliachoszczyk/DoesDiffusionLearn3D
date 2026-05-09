$blender    = "C:\Program Files (x86)\blender-2.78c-windows64\blender.exe"
$script     = "D:\STUDIA SZI\Semestr 2\Zaawansowane zagadnienia sieci neuronowych\Projekt\repo\DoesDiffusionLearn3D\clevr-dataset-gen\image_generation\render_images.py"
$base_out   = "D:\STUDIA SZI\Semestr 2\Zaawansowane zagadnienia sieci neuronowych\Projekt\repo\DoesDiffusionLearn3D\data\v16"
$clevr_data = "D:\STUDIA SZI\Semestr 2\Zaawansowane zagadnienia sieci neuronowych\Projekt\repo\DoesDiffusionLearn3D\clevr-dataset-gen\image_generation\data"

$shapes = @("SmoothCube_v2", "Sphere", "SmoothCylinder")
$images_per_shape = 3
$az_range = @(0.0, 360.0)
$el_range = @(30.0, 75.0)


foreach ($shape in $shapes) {
    $out_img   = "$base_out\images\$shape"
    $out_scene = "$base_out\scenes\$shape"
    New-Item -ItemType Directory -Force -Path $out_img   | Out-Null
    New-Item -ItemType Directory -Force -Path $out_scene | Out-Null

    Write-Host "Rendering $images_per_shape images for shape=$shape"

    & $blender --background -noaudio --python $script -- `
        --num_images $images_per_shape `
        --render_num_samples 64 `
        --shape $shape `
        --start_idx 0 `
        --output_image_dir $out_img `
        --output_scene_dir $out_scene `
        --filename_prefix "$shape" `
        --width 512 `
        --height 512 `
        --key_light_jitter 0 `
        --fill_light_jitter 0 `
        --back_light_jitter 0 `
        --az_range $az_range[0] $az_range[1] `
        --el_range $el_range[0] $el_range[1] `
        --base_scene_blendfile "$clevr_data\base_scene.blend" `
        --properties_json "$clevr_data\properties.json" `
        --shape_dir "$clevr_data\shapes" `
        --material_dir "$clevr_data\materials" `
        --output_scene_file "$out_scene\CLEVR_scenes.json"
}
Write-Host "Done!"