#########Code Contributions#######
#  command to run by using camera - py yolo_opencv.py -s true --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
#  command to run by taking images from file - py yolo_opencv.py -s false --image train.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt

# importing all the required packages
from cgitb import text
import cv2
import traceback
import argparse
import numpy as np
from gtts import gTTS
from playsound import playsound

image_filename = None

# a class for argument parsing
class argument_parser_class:
    def __init__(self):
        self.argument_parser = argparse.ArgumentParser()
    
    def add_arguments(self):
        self.argument_parser.add_argument('-s', '--capture', required=True,
                help = 'Yes or No for image capturing')
        self.argument_parser.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
        self.argument_parser.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
        self.argument_parser.add_argument('-i', '--image', required=False,
                help = 'path to input image')
        self.argument_parser.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')

class utility:
    def __init__(self):
        pass
    
    # output layer
    def get_output_layers(self,net):
        output  = []
        layer_names = net.getLayerNames()
        ran = net.getUnconnectedOutLayers()
        for i in ran:
            output.append(layer_names[i-1])
        return output

    # bounding boxes
    def draw_predictions(self, label, len_classes, img, id , x, y, x_p_w, y_p_h):
        COLORS = np.random.uniform(0, 255, size=(len_classes, 3))
        cv2.rectangle(img, (x,y), (x_p_w,y_p_h), COLORS[id], 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[id], 2)

class get_cv2_attr():
    def __init__(self):
        self.scale = 0.00392

    # loading pre trained weights and config gile
    def get_net(self,args):
        net = cv2.dnn.readNet(args.weights, args.config)
        return net

    # creating blob
    def get_blob(self):
        blob = cv2.dnn.blobFromImage(image,self.scale,(416,416),(0,0,0), True, crop=False)
        return blob

    def get_outs(self,args):
        net = self.get_net(args)
        blob = self.get_blob()

        # setting blob for network
        net.setInput(blob)
        outs = net.forward(utility_.get_output_layers(net))
        return outs

# creating an instance and adding arguments, parse args
argument_parser = argument_parser_class()
argument_parser.add_arguments()
args = argument_parser.argument_parser.parse_args()

utility_ = utility()


if (args.capture == "true"):
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)

            key = cv2.waitKey(1)
            if key == ord('s'): 
                img_new = cv2.imwrite(filename='saved_img.jpg', img=frame)
                image_filename = 'saved_img.jpg'
                webcam.release()
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
            
                break
            elif key == ord('q'):
                webcam.release()
                cv2.destroyAllWindows()
                break
            
        except(KeyboardInterrupt):
            #Turns off camera
            webcam.release()
            cv2.destroyAllWindows()
            break

if (args.capture == "false"):
    image_filename =  args.image  

image = cv2.imread(image_filename)

Width = image.shape[1]

Height = image.shape[0]
# scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

cv2_attr = get_cv2_attr()
outs = cv2_attr.get_outs(args)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            class_ids.append(class_id)
            confidences.append(float(confidence))
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

text1 = "The objects in the image are: "

for i in indices:
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    utility_.draw_predictions(str(classes[class_ids[i]]), len(classes), image, class_ids[i], round(x), round(y), round(x+w), round(y+h))
    fp = open("./yolov3.txt")
    for j, line in enumerate(fp):
        if j == class_ids[i]:
            text_1 = line.strip()
            text1 += (text_1+",")
    fp.close()

myobj = gTTS(text=text1, lang='en', slow=False)
  
myobj.save("sarat.mp3")


cv2.imshow("object detection", image)
playsound("sarat.mp3")
cv2.waitKey(5000)
    
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()