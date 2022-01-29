import cv2 

class NMS:
    def __init__(self) -> None:
        self.conf = 0.5
        self.iou_threshsold = 0.4

    def draw_overlay(self, image, bboxes_list):
        overlay_color = {
            '0' : (0, 255, 0),
            '1' : (255, 0, 0)
        }
        overlay_thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        for coord in bboxes_list:
            class_name = coord[0]
            start_point = (int(coord[1]), int(coord[2]))
            end_point = (int(coord[3]), int(coord[4]))
            prob = float(coord[5])
            text_start_point = (int(coord[1]), int(coord[2]) - 10)
            
            image = cv2.rectangle(image, start_point, end_point, \
                    overlay_color[class_name], overlay_thickness)
            image = cv2.putText(image, str(prob), text_start_point, \
                font, 0.8, overlay_color[class_name], overlay_thickness - 1, cv2.LINE_AA)
        
        cv2.imshow("im", image) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 


    def IOU(self, bboxes1, bboxes2):
        bboxes1 = [int(i) for i in bboxes1]
        bboxes2= [int(i) for i in bboxes2]

        xA = max(bboxes1[0], bboxes2[0])
        yA = max(bboxes1[1], bboxes2[1])
        xB = min(bboxes1[2], bboxes2[2])
        yB = min(bboxes1[3], bboxes2[3])

        intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        box1_area = (bboxes1[2] - bboxes1[0] + 1) * (bboxes1[3] - bboxes1[1] + 1)
        box2_area = (bboxes2[2] - bboxes2[0] + 1) * (bboxes2[3] - bboxes2[1] + 1)

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
 
        return iou

    def nms(self, image, bboxes_list):
        req_bboxes, final_boxes = [], []

        for coord in bboxes_list: 
            prob = float(coord[5])
            if prob > self.conf:
                req_bboxes.append(coord)

        # sorting the bounding boxes based on probability score
        bboxes_sorted = sorted(req_bboxes, reverse=True, key=lambda x: x[5])

        while len(bboxes_sorted) > 0:
            # removing the best probability bounding box
            box = bboxes_sorted.pop(0)

            for b in bboxes_sorted:
                # comparing with the same class
                if box[0] == b[0]:
                    iou = self.IOU(box[1:-1], b[1:-1])
                    if iou >= self.iou_threshsold:
                        # if IOU is large then discard the box with lowest probability
                        bboxes_sorted.remove(b)
            print(len(bboxes_sorted))

            final_boxes.append(box)

        return final_boxes


if __name__ == "__main__":
    obj = NMS()
    image = cv2.imread("zoraya.jpg")


    with open("coordinates.txt", 'r') as f:
        data = f.readlines()
    data = [i[:-1].split(' ') for i in data]


    obj.draw_overlay(image, data)
    final_boxes = obj.nms(image, data)
    obj.draw_overlay(image, final_boxes)