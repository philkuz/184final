import neural_style as ns
# defaults in param_dict
dirname = lambda x: '/'.join(x.split('/')[:-1])
filename = lambda x: x.split('/')[-1]
content_img = '../data/geometry-v1/no_texture/0000000.png'
param_dict = {
    'device': '/gpu:0',
    'content_dir' : dirname(content_img),
    'content_filename' : filename(content_img)
}

styles_to_try = ['../data/geometry-v1/texture/0000000.png', '../data/geometry-v1/texture/0000001.png', '../data/geometry-v1/texture/0000002.png']
style_weights_to_try = [1e4, 1e3, 1e2, 1e5]
content_weights_to_try = [5e0, 5e-1, 5e1]
style_weights_to_try = style_weights_to_try[:1]
content_weights_to_try = content_weights_to_try[:2]
styles_to_try = styles_to_try[1:]



'../data/geometry-v1/texture/0000000.png'
ipt_args = ["--content_img",
  "{content_filename}",
	"--content_img_dir",
	 "{content_dir}",
	"--style_imgs",
	 "{style_filename}",
	"--style_imgs_dir",
	 "{style_dir}",
	"--device",
	 "{device}",
	"--img_name",
	 "{output_image}",
	"--style_weight",
  "{style_weight}",
  "--content_weight",
  "{content_weight}"]
args_joined = ' '.join(ipt_args)
for style_w in style_weights_to_try:
  for content_w in content_weights_to_try:
    for style in styles_to_try:
      param_dict['style_filename'] = filename(style)
      param_dict['style_dir'] = dirname(style)
      param_dict['style_weight'] = style_w
      param_dict['content_weight'] = content_w
      param_dict['output_image'] = 'content={},style={},file={}'.format(content_w, style_w, filename(style).split('.')[0])

      args_formatted = args_joined.format(**param_dict).split(' ')
      print(args_formatted)
      ns.run_image(args_formatted)

