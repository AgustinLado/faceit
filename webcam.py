"""
Lightly modified version of the face_recognition webcam tutorials.
"""

import cv2
import face_recognition

# For performance
SCALE = 4

# Get a reference to webcam
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
al_image = face_recognition.load_image_file('dataset/dudes/al.jpg')
al_face_encoding = face_recognition.face_encodings(al_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    al_face_encoding,
]
known_face_names = [
    'Agustin',
]

while True:
    # Grab a single frame of video
    original_frame = video_capture.read()[1]
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(original_frame, (0, 0),
                             fx=1 / SCALE, fy=1 / SCALE)
    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face matches a known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = '?'

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Scale the face locations back up
        top *= SCALE
        right *= SCALE
        bottom *= SCALE
        left *= SCALE
        # Draw a box around the face
        cv2.rectangle(original_frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(original_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 3)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(original_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', original_frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
