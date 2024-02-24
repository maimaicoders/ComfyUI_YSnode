# only import if running as a custom node
try:
	import comfy.utils
except ImportError:
	pass
else:
	NODE_CLASS_MAPPINGS = {}
	from .nodes.images import NODE_CLASS_MAPPINGS as ImgNodes
	NODE_CLASS_MAPPINGS.update(ImgNodes)

	from .nodes.establish_node import NODE_CLASS_MAPPINGS as openpose_node
	NODE_CLASS_MAPPINGS.update(openpose_node)

	NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
