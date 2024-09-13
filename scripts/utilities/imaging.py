import numpy as np

class ImageUtilities:

    @staticmethod
    def find_color(rgb : np.array, image : np.array):
        if len(rgb) != 3:
            raise ValueError("Input a valid rgb value as numpy array with three values.")
        if len(image.shape) != 3:
            raise ValueError("Input a colored 2D image as numpy array.")

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                #print([image[i,j,0], image[i,j,1], image[i,j,2]])
                if image[i, j, 0] == rgb[0] and image[i, j, 1] == rgb[1] and image[i,j,2] == rgb[2]:
                    return [i,j]
                
        return None
