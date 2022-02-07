import pixellib
from pixellib.instance import instance_segmentation as iseg

def get_extended_image(img, x, y, w, h, k=0.1):
    '''
    Function, that return cropped image from 'img'
    If k=0 returns image, cropped from (x, y) (top left) to (x+w, y+h) (bottom right)
    If k!=0 returns image, cropped from (x-k*w, y-k*h) to (x+k*w, y+(1+k)*h)
    
    Parameters:
        img : The original image
        x : x coordinate of the upper-left corner
        y : y coordinate of the upper-left corner
        w : Width of the desired image
        h : Height of the desired image
        k : The coefficient of expansion of the image

    Returns:
        image (resized mage with extra dimension at axis=0 as a numpy array)
    '''

    # The next code block checks that coordinates will be non-negative
    # (in case if desired image is located in top left corner)
    if x - k*w > 0:
        start_x = int(x - k*w)
    else:
        start_x = x
    if y - k*h > 0:
        start_y = int(y - k*h)
    else:
        start_y = y

    end_x = int(x + (1 + k)*w)
    end_y = int(y + (1 + k)*h)

    face_image = img[start_y:end_y,
                    start_x:end_x]

    face_image = cv2.resize(face_image, (224, 224))
    face_image = face_image/255

    # shape from (250, 250, 3) to (1, 250, 250, 3)
    face_image = np.expand_dims(face_image, axis=0)
    
    return face_image


def apply_mask(image,model):
    segment_image = iseg()
    segment_image.load_model(model)
    result = segment_image.segmentFrame(image,show_bboxes=False)
    return result