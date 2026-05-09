$blender    = "C:\Program Files (x86)\blender-2.78c-windows64\blender.exe"
$script     = "D:\STUDIA SZI\Semestr 2\Zaawansowane zagadnienia sieci neuronowych\Projekt\repo\DoesDiffusionLearn3D\clevr-dataset-gen\image_generation\render_images.py"
$base_out   = "D:\STUDIA SZI\Semestr 2\Zaawansowane zagadnienia sieci neuronowych\Projekt\repo\DoesDiffusionLearn3D\data\v12"
$clevr_data = "D:\STUDIA SZI\Semestr 2\Zaawansowane zagadnienia sieci neuronowych\Projekt\repo\DoesDiffusionLearn3D\clevr-dataset-gen\image_generation\data"

$shapes = @("SmoothCube_v2", "Sphere", "SmoothCylinder")
$images_per_shape = 100

foreach ($shape in $shapes) {
    $out_img   = "$base_out\images\$shape"
    $out_scene = "$base_out\scenes\$shape"
    New-Item -ItemType Directory -Force -Path $out_img   | Out-Null
    New-Item -ItemType Directory -Force -Path $out_scene | Out-Null

    for ($i = 0; $i -lt $images_per_shape; $i++) {
        $idx = $shapes.IndexOf($shape) * $images_per_shape + $i
        $az  = [math]::Round((Get-Random -Minimum 0.0  -Maximum 360.0), 1)
        $el  = [math]::Round((Get-Random -Minimum 30.0 -Maximum 75.0),  1)

        Write-Host "[$idx] shape=$shape az=$az el=$el"

        & $blender --background -noaudio --python $script -- `
            --num_images 1 `
            --render_num_samples 128 `
            --shape $shape `
            --light_azimuth $az `
            --light_elevation $el `
            --start_idx $idx `
            --output_image_dir $out_img `
            --output_scene_dir $out_scene `
            --filename_prefix "LIGHT_v7" `
            --width 512 `
            --height 512 `
            --key_light_jitter 0 `
            --fill_light_jitter 0 `
            --back_light_jitter 0 `
            --base_scene_blendfile "$clevr_data\base_scene.blend" `
            --properties_json "$clevr_data\properties.json" `
            --shape_dir "$clevr_data\shapes" `
            --output_scene_file "$out_scene\CLEVR_scenes.json" `
            --material_dir "$clevr_data\materials"
    }
}
Write-Host "Gotowe!"