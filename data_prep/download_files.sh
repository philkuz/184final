mkdir -p ../data/geometry-v1
cd ../data/geometry-v1
# src textures
gdrive download 1X8QFfBeRgGEzPWMHKfjmVQlKktLx6CuG
unzip textures.zip
rm -rf __MACOSX textures.zip
mv textures src_textures
# no-texture image
gdrive download 1Mu2mF3qxqODx7cwSwYvtdZd8qQssvqkF
# textured images
gdrive download 17U9JRVDXpqq811CB_crN68drs_Oi5NGJ
unzip output.zip
rm -rf __MACOSX output.zip
mv output texture

# unzipping
mkdir no_texture
mv theplane.png no_texture/0000000.png
