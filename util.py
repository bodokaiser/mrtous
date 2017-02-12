from skimage.util.shape import view_as_windows

def image_to_patches(image, size):
    if np.shape(image)[0] == 1:
        image = image[0]
    patches = view_as_windows(image, size, size)
    patches = np.reshape(patches, [-1, 1, size, size])
    return patches

def filter_patches(patches, value=.0001):
    return patches[np.sum(patches**2, (1, 2, 3)) > value]