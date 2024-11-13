# imports necessary computer vision, computational, and machine learning libraries
import cv2
import face_recognition

# stores photo files in respective variables
image = face_recognition.load_image_file('Photos/Oliver.jpg')       # training image
image2 = face_recognition.load_image_file('Photos/Oliver2.jpg')
image3 = face_recognition.load_image_file('Photos/Oliver3.jpg')
image4 = face_recognition.load_image_file('Photos/Oliver4.jpg')

# converts images from BGR (blue, green, red) to RGB (red, green, blue)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)

# contains coordinates of the face in the image
faceLoc = face_recognition.face_locations(image)[0]
faceLoc2 = face_recognition.face_locations(image2)[0]
faceLoc3 = face_recognition.face_locations(image3)[0]
faceLoc4 = face_recognition.face_locations(image4)[0]

# stores the face in a numerical vectors
encode = face_recognition.face_encodings(image)[0]
encode2 = face_recognition.face_encodings(image2)[0]
encode3 = face_recognition.face_encodings(image3)[0]
encode4 = face_recognition.face_encodings(image4)[0]

# adds a visual border representation around the detected face
cv2.rectangle(image, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 3)
cv2.rectangle(image2, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 255), 3)
cv2.rectangle(image3, (faceLoc3[3], faceLoc3[0]), (faceLoc3[1], faceLoc3[2]), (255, 0, 255), 3)
cv2.rectangle(image4, (faceLoc4[3], faceLoc4[0]), (faceLoc4[1], faceLoc4[2]), (255, 0, 255), 3)

# computes whether the faces detected are the same and returns a boolean value
compRes1 = face_recognition.compare_faces([encode], encode2)
compRes2 = face_recognition.compare_faces([encode], encode3)
compRes3 = face_recognition.compare_faces([encode], encode4)

# computes the Euclidean distance to determine how distinct the two faces are
compDis1 = face_recognition.face_distance([encode], encode2)
compDis2 = face_recognition.face_distance([encode], encode3)
compDis3 = face_recognition.face_distance([encode], encode4)

# adds the results of the comparison in the form of text onto the displayed images
cv2.putText(image2, f'Match = {compRes1[0]}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)
cv2.putText(image3, f'Match = {compRes2[0]}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)
cv2.putText(image4, f'Match = {compRes3[0]}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)

# adds the results of the comparison distance in the form of text onto the displayed images
cv2.putText(image2, f'Distance = {round(compDis1[0],4)}', (50,80), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)
cv2.putText(image3, f'Distance = {round(compDis2[0],4)}', (50,80), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)
cv2.putText(image4, f'Distance = {round(compDis3[0],4)}', (50,80), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 2)

# prints the results of the comparison to the console as well
print(f'Test 1: {compRes1[0]} {compDis1[0]}')
print(f'Test 2: {compRes2[0]} {compDis2[0]}')
print(f'Test 3: {compRes3[0]} {compDis3[0]}')

# displays the gui of the image to the user
cv2.imshow('Oliver', image)
cv2.imshow('Oliver Test 1', image2)
cv2.imshow('Oliver Test 2', image3)
cv2.imshow('Oliver Test 3', image4)

# puts the program on hold so that the images stay displayed
cv2.waitKey(0)