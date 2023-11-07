import cv2
import numpy as np

def get_center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def detect_people_yolo(video_path, yolo_cfg, yolo_weights, coco_names):
    frame_id = 0
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    layer_names = net.getLayerNames()
    out_layers_indices = net.getUnconnectedOutLayers()

    if out_layers_indices.ndim == 2:
        out_layers_indices = out_layers_indices[:, 0]

    output_layers = [layer_names[i - 1] for i in out_layers_indices]
    classes = open(coco_names).read().strip().split("\n")

    cap = cv2.VideoCapture(video_path)

    prev_boxes = []
    prev_ids = []
    next_id = 1

    down = 0
    up = 0
    down_ids = set()
    up_ids = set()
    prev_centers = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "person":
                    center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

        if len(indexes) > 0 and isinstance(indexes[0], list):
            indexes = [i[0] for i in indexes]

        current_boxes = [boxes[i] for i in indexes]
        current_ids = [-1] * len(current_boxes)

        for i, box in enumerate(current_boxes):
            center_current = get_center(box)

            best_dist = float('inf')
            best_id = -1

            for j, prev_box in enumerate(prev_boxes):
                center_prev = get_center(prev_box)
                dist = get_distance(center_current, center_prev)

                if dist < best_dist:
                    best_dist = dist
                    best_id = prev_ids[j]

            if best_dist > 30 or best_id == -1:
                current_ids[i] = next_id
                next_id += 1
            else:
                current_ids[i] = best_id

        # Adiciona IDs de pessoas aos quadros anteriores
        prev_boxes = current_boxes
        prev_ids = current_ids

        for i in range(len(current_boxes):
            x, y, w, h = current_boxes[i]
            center = get_center(current_boxes[i])
            person_id = current_ids[i]

            if person_id not in prev_centers:
                prev_centers[person_id] = []

            prev_centers[person_id].append(center)

            if len(prev_centers[person_id]) > 4:
                prev_centers[person_id].pop(0)

            if len(prev_centers[person_id]) > 2:
                move_up = True
                move_down = True

                for j in range(1, len(prev_centers[person_id])):
                    if prev_centers[person_id][j][1] > prev_centers[person_id][j - 1][1]:
                        move_up = False
                    if prev_centers[person_id][j][1] < prev_centers[person_id][j - 1][1]:
                        move_down = False

                if move_up and person_id not in up_ids:
                    up += 1
                    up_ids.add(person_id)
                elif move_down and person_id not in down_ids:
                    down += 1
                    down_ids.add(person_id)

            label = f"Person {person_id}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (int(center[0]), int(center[1]), 2, (255, 255, 0), -2)

        cv2.line(frame, (0, 300), (frame.shape[1], 300), (0, 0, 255), 1)
        cv2.imshow("YOLO People Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Pessoas que se moveram para baixo: {down}")
    print(f"Pessoas que se moveram para cima: {up}")

video_path = 'C:\contadorPessoas\video_cortado.mp4'

yolo_cfg = 'C:\contadorPessoas\1 - testeBiblios\YOLOtest\yolov3.cfg'
yolo_weights= 'C:\contadorPessoas\1 - testeBiblios\YOLOtest\yolov3.weights'
coco_names = = 'C:\contadorPessoas\1 - testeBiblios\YOLOtest\coco.names'

detect_people_yolo(video_path, yolo_cfg, yolo_weights, coco_names)
