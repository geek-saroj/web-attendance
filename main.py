import os
import cv2
import face_recognition
import csv
import datetime

# Define list of supported image formats
IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

# Load known faces and their names from images in the directory
known_faces_names = []
known_faces_encodings = {}

for filename in os.listdir('known_faces'):
    # Check if the file is an image
    if os.path.splitext(filename)[1].lower() not in IMAGE_FORMATS:
        continue

    # Load image and encoding
    image = face_recognition.load_image_file(os.path.join('known_faces', filename))
    encoding = face_recognition.face_encodings(image)[0]
    name = os.path.splitext(filename)[0]
    known_faces_names.append(name)
    known_faces_encodings[name] = encoding

# Initialize webcam
cap = cv2.VideoCapture(1)

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
present_students = []
last_attendance_time = {}

while True:
    # Capture image from webcam
    ret, frame = cap.read()

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the current frame
    for face_encoding in face_encodings:
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(list(known_faces_encodings.values()), face_encoding)
        name = "Unknown"

        # Find the index of the first match
        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_faces_encodings.keys())[first_match_index]
            present_students.append(name)

        # Write data to CSV file
        with open('attendance.csv', mode='a') as file:
            writer = csv.writer(file)
            if os.stat('attendance.csv').st_size == 0:
                writer.writerow(['Name', 'Date', 'Time'])
            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            # Check if the student's attendance was already recorded today
            if name in last_attendance_time and (now - last_attendance_time[name]).days == 0:
                continue

            # Write the attendance record and update the last attendance time
            writer.writerow([name, date, time])
            last_attendance_time[name] = now

        # Display the name on the image
        cv2.rectangle(frame, (0, 0), (500, 50), (0, 0, 0), -1)
        cv2.putText(frame, name +  "Present", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting image
    cv2.imshow('Attendance System', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
