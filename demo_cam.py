import cv2


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



faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
#region
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'

genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'
#endregion


faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

age_list = ['(0-8)', '(9-14)', '(15-17)', '(17-19)', '(20-24)','(23-30)', '(38-53)', '(54-59)', '(60-100)']
gender_list = ['Male', 'Female']
mean_value = (78.4263377603, 87.7689143744, 114.895847746)


padding = 0
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)
   
    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)  # Normalize the brightness and contrast
    for bbox in bboxs:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        try:
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), mean_value, swapRB=False) #Feeding the system with the face image, edited to 227 × 227 around the face focus.  
        except Exception as e:
            print(str(e))

        genderNet.setInput(blob)
        gender_pred = genderNet.forward()
        gender = gender_list[gender_pred[0].argmax()]

        ageNet.setInput(blob)
        age_pred = ageNet.forward()
        age = age_list[age_pred[0].argmax()]
        print(gender)
        label = "{},{}".format(gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1]-10),
                      (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Age and Gender Predictor Project', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
