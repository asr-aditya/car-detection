import numpy as np
import cv2

class bb:
    def __init__(self):
        self.x, self.y, self.h, self.w, self.c, self.prob, self.x1, self.y1, self.x2, self.y2\
            = None, None, None, None, None, None, None, None, None, None

#read weights from file and load them into the model
def load_weights(model,yolo_weight_file):
    weights = np.fromfile(yolo_weight_file,np.float32)
    weights = weights[4:]
    
    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            shape_kernal,shape_bias = shape
            bias = weights[index:index+np.prod(shape_bias)].reshape(shape_bias)
            index += np.prod(shape_bias)
            kernal = weights[index:index+np.prod(shape_kernal)].reshape(shape_kernal)
            index += np.prod(shape_kernal)
            layer.set_weights([kernal,bias])


#change yolo output into box values that corrospond to the original image.
def process_output(yolo_output, threshold=0.2, padhw=(98,0), shaved=False, shavedim=(350,500, 500,1000)):
    # Class label for car in the dataset
    car_class = 6
    boxes = []
    S = 7
    B = 2
    C = 20
    SS = S * S  # num yolo grid cells
    prob_size = SS * C  # num class probabilities
    conf_size = SS * B  # num confidences, 2 per grid cell

    probs = yolo_output[0:prob_size]  # seperate probability array
    confidences = yolo_output[prob_size:prob_size + conf_size]  # seperate confidence array
    yolo_boxes = yolo_output[prob_size + conf_size:]  # seperate coordinates

    # reshape arrays so that each cell in the yolo grid is a seperate array containing the cells properties
    probs = np.reshape(probs, (SS, C))
    confs = np.reshape(confidences, (SS, B))
    yolo_boxes = np.reshape(yolo_boxes, (SS, B, 4))

    # itterate through grid and then boxes in each cell
    gridn = 0
    for gridy in range(S):
        for gridx in range(S):
            for index1 in range(B):

                box = bb()
                box.c = confs[gridn, index1]
                p = probs[gridn, :] * box.c

                if (p[car_class] >= threshold):
                    print("Yes the given image is a car")
                else:
                    print("The given image is not a car")


                    