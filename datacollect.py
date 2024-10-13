import cv2 
import os

video = cv2.VideoCapture(0)

user_name = input("Enter User Name: ")
user_folder = os.path.join('datasets', user_name)
os.makedirs(user_folder, exist_ok=True)  # Créer un dossier pour l'utilisateur s'il n'existe pas

count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Enregistrez l'image couleur dans le dossier de l'utilisateur
    image_path = os.path.join(user_folder, f'{user_name}.{count}.jpg')
    cv2.imwrite(image_path, frame)
    
    count = count + 1

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if count > 100:
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done..................")
