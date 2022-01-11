import cv2
import imutils
import numpy as np
import random
import colorsys


def get_random_bright_colors(size):
    for i in range(0, size):
        h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
        r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
        yield (r, g, b)


def show(bboxes, boxes, frame, labels, classIDs, confidences):

    colors = list(get_random_bright_colors(len(labels)))

    if len(bboxes) > 0:
        for i in bboxes.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x - w/2, y - h/2), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])

            # draw bounding box title background
            text_offset_x = x
            text_offset_y = y
            text_color = (255, 255, 255)
            (text_width, text_height) = \
                cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=1)[0]
            box_coords = (
                (text_offset_x, text_offset_y),
                (text_offset_x + text_width - 80, text_offset_y - text_height + 4))
            cv2.rectangle(frame, box_coords[0], box_coords[1], color, cv2.FILLED)

            # draw bounding box title
            cv2.putText(frame, text, (x - w/2, y - h/2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return
