import os
import pickle

import cv2
import face_recognition

BASE_DIR = 'dataset/jurassic_park'
INPUT_FILENAME = f'{BASE_DIR}/lunch_scene.mp4'
OUTPUT_FILENAME = f'{BASE_DIR}/output.mp4'

KNOWN_FACES_PATH = 'known_faces.pkl'

KNOWN_FACES = {}
TOLERANCE = 0.6


def get_known_faces():
    """
    Get a dictionary in the form of {name: [face_encodings]} representing the
    known faces.
    """
    if os.path.exists(KNOWN_FACES_PATH):
        print(f'Getting known_faces from "{KNOWN_FACES_PATH}"')
        with open(KNOWN_FACES_PATH, 'rb') as f:
            return pickle.loads(f.read())

    print('Populating KNOWN_FACES')
    faces = {name: [] for name in next(os.walk(BASE_DIR))[1]}
    for name, encodings in faces.items():
        face_dir = os.path.join(BASE_DIR, name)
        for image in next(os.walk(face_dir))[2]:
            print(f'\tDetecting {os.path.join(name, image)}')
            face_image = face_recognition.load_image_file(
                os.path.join(face_dir, image))
            detected = face_recognition.face_encodings(face_image)
            if detected:
                encodings.append(detected[0])

    print(f'Saving KNOWN_FACES to {KNOWN_FACES_PATH}')
    with open(KNOWN_FACES_PATH, 'wb') as f:
        f.write(pickle.dumps(faces))

    return faces


def resolve_face(face_to_compare):
    """
    Compare the face with all the known faces.
    Return the one with the lowest distance if it's below the tolerance.
    TODO: Try out a neural network :D
    """
    distance_averages = {}
    for name, encodings in KNOWN_FACES.items():
        distances = face_recognition.face_distance(encodings, face_to_compare)
        distance_averages[name] = sum(distances) / len(distances)

    # Get the known face closest to the face to compare
    closest_face = min(distance_averages.items(), key=lambda x: x[1])
    # Only the distances under the tolerance threshold are considered matches
    if closest_face[1] < TOLERANCE:
        return closest_face[0]
    return '?'


def label_frame(frame, faces):
    """ Label the results """
    for (top, right, bottom, left), name in faces:
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom),
                      (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


def main():
    # Open the input video file
    input_video = cv2.VideoCapture(INPUT_FILENAME)
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create the output video file
    # TODO: Grab the resolution and framerate with mediainfo
    output_video = cv2.VideoWriter(
        OUTPUT_FILENAME,
        cv2.VideoWriter_fourcc(*'mp4v'),
        23.976,
        (640, 352),
    )

    counter = 0

    while True:
        # Grab a single frame of video
        ret, original_frame = input_video.read()
        # Quit when the input video file ends
        if not ret:
            break
        counter += 1
        if counter % 100 == 0:
            print(f'Processing frame #{counter} out of {total_frames}')

        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        frame = original_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Recognize the faces in the video
        face_names = [resolve_face(encoding) for encoding in face_encodings]
        # Label the frame
        label_frame(original_frame, zip(face_locations, face_names))

        # Display the frame
        cv2.imshow('Jurassic Park', original_frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write the resulting image to the output video file
        output_video.write(original_frame)

    print('All done!')
    input_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Hi!')
    KNOWN_FACES = get_known_faces()
    main()
