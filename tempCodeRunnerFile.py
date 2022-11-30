def faceBox(faceNet, frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (200, 200), [
                                 104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3]*frame_width)
            y1 = int(detection[0, 0, i, 4]*frame_height)
            x2 = int(detection[0, 0, i, 5]*frame_width)
            y2 = int(detection[0, 0, i, 6]*frame_height)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs