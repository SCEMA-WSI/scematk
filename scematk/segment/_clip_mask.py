from ..image._binary_mask import BinaryMask
from ..image._image import Image
from ..image._label_mask import LabelMask
from ..image._ubyte_image import UByteImage


def clip_mask(image: Image, mask: BinaryMask) -> Image:
    assert isinstance(image, Image), "image must be an Image"
    assert isinstance(mask, BinaryMask), "mask must be a BinaryMask"
    assert image.shape[:2] == mask.shape[:2], "image and mask must have the same shape"
    img = image.image
    mask = mask.image
    img = img * mask
    if isinstance(image, BinaryMask):
        return BinaryMask(img, image.info, image.channel_names)
    elif isinstance(image, LabelMask):
        return LabelMask(img, image.info, image.channel_names)
    elif isinstance(image, UByteImage):
        return UByteImage(img, image.info, image.channel_names)
    else:
        raise NotImplementedError(f"Images of type {type(image)} are not supported yet.")
