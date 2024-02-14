import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
import threading

def resize(image, width=None, height=None):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def rotate_image(image, angle):

    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def apply_filter(image, filter_type):

    if filter_type == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_type == 'edge':
        return cv2.Canny(image, 50, 150)
    else:
        return image

def concatenate_images_vertically(image1, image2):

    return np.concatenate((image1, image2), axis=0)

def concatenate_images_horizontally(image1, image2):

    return np.concatenate((image1, image2), axis=1)

def capture_single_side():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the rectangular bounding area
    rect_start_point = (int(width/4), int(height/4))
    rect_end_point = (int(3*width/4), int(3*height/4))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        # Draw the rectangular bounding area on the frame
        cv2.rectangle(frame, rect_start_point, rect_end_point, (0, 255, 0), 2)

        cv2.imshow('Capture Single Side', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Capture the ID card within the rectangular bounding area
            id_card = frame[rect_start_point[1]:rect_end_point[1], rect_start_point[0]:rect_end_point[0]]

            # Resize captured ID card
            id_card = resize(id_card, width=400)

            cv2.imshow('Captured Single Side', id_card)
            cv2.imwrite('single_side.jpg', id_card)
            messagebox.showinfo("Success", "Single side captured as 'single_side.jpg'.")
            break

    cap.release()
    cv2.destroyAllWindows()
    add_rotation_and_filter_options()

def capture_first_side():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the rectangular bounding area
    rect_start_point = (int(width/4), int(height/4))
    rect_end_point = (int(3*width/4), int(3*height/4))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        # Draw the rectangular bounding area on the frame
        cv2.rectangle(frame, rect_start_point, rect_end_point, (0, 255, 0), 2)

        cv2.imshow('Capture First Side', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Capture the ID card within the rectangular bounding area
            id_card = frame[rect_start_point[1]:rect_end_point[1], rect_start_point[0]:rect_end_point[0]]

            # Resize captured ID card
            id_card = resize(id_card, width=400)

            cv2.imshow('Captured First Side', id_card)
            cv2.imwrite('first_side.jpg', id_card)
            messagebox.showinfo("Success", "First side captured as 'first_side.jpg'.")
            break

    cap.release()
    cv2.destroyAllWindows()
    add_rotation_and_filter_options()

def capture_second_side():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the rectangular bounding area
    rect_start_point = (int(width/4), int(height/4))
    rect_end_point = (int(3*width/4), int(3*height/4))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        # Draw the rectangular bounding area on the frame
        cv2.rectangle(frame, rect_start_point, rect_end_point, (0, 255, 0), 2)

        cv2.imshow('Capture Second Side', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Capture the ID card within the rectangular bounding area
            id_card = frame[rect_start_point[1]:rect_end_point[1], rect_start_point[0]:rect_end_point[0]]

            # Resize captured ID card
            id_card = resize(id_card, width=400)

            cv2.imshow('Captured Second Side', id_card)
            cv2.imwrite('second_side.jpg', id_card)
            messagebox.showinfo("Success", "Second side captured as 'second_side.jpg'.")
            break

    cap.release()
    cv2.destroyAllWindows()
    concatenate_and_process()

def add_rotation_and_filter_options():
    rotate_label.pack()
    rotate_90_button.pack()
    rotate_180_button.pack()
    rotate_270_button.pack()

    filter_label.pack()
    grayscale_button.pack()
    blur_button.pack()
    edge_detection_button.pack()

    crop_button.pack()

def concatenate_and_process():
    first_side_image = cv2.imread('first_side.jpg')
    second_side_image = cv2.imread('second_side.jpg')

    concatenated_image = concatenate_images_vertically(first_side_image, second_side_image)
    cv2.imwrite('concatenated_id_card.jpg', concatenated_image)
    messagebox.showinfo("Success", "Concatenated ID card saved as 'concatenated_id_card.jpg'.")
    add_rotation_and_filter_options()

def crop_manually():
    messagebox.showinfo("Manual Crop", "Please draw a rectangle around the area you want to crop, then press 'c' to confirm.")

    image = cv2.imread('concatenated_id_card.jpg')
    clone = image.copy()
    rect_start_point = None
    rect_end_point = None
    cropping = False

    def mouse_click(event, x, y, flags, param):
        nonlocal rect_start_point, rect_end_point, cropping

        if event == cv2.EVENT_LBUTTONDOWN:
            rect_start_point = (x, y)
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            rect_end_point = (x, y)
            cropping = False
            cv2.rectangle(clone, rect_start_point, rect_end_point, (0, 255, 0), 2)
            cv2.imshow("Concatenated ID Card", clone)

    cv2.namedWindow("Concatenated ID Card")
    cv2.setMouseCallback("Concatenated ID Card", mouse_click)

    while True:
        cv2.imshow("Concatenated ID Card", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if rect_start_point and rect_end_point:
                cropped_id_card = clone[min(rect_start_point[1], rect_end_point[1]):max(rect_start_point[1], rect_end_point[1]),
                                       min(rect_start_point[0], rect_end_point[0]):max(rect_start_point[0], rect_end_point[0])]
                cv2.imwrite('cropped_id_card.jpg', cropped_id_card)
                messagebox.showinfo("Success", "Cropped ID card saved as 'cropped_id_card.jpg'.")
                break

    cv2.destroyAllWindows()

def on_rotate_click(angle):
    threading.Thread(target=rotate_and_save, args=(angle,)).start()

def on_filter_click(filter_type):
    threading.Thread(target=apply_filter_and_save, args=(filter_type,)).start()

def rotate_and_save(angle):
    image = cv2.imread('concatenated_id_card.jpg')
    rotated_image = rotate_image(image, angle)
    cv2.imwrite('rotated_id_card.jpg', rotated_image)
    messagebox.showinfo("Success", "Rotated ID card saved as 'rotated_id_card.jpg'.")

def apply_filter_and_save(filter_type):
    image = cv2.imread('concatenated_id_card.jpg')
    filtered_image = apply_filter(image, filter_type)
    cv2.imwrite('filtered_id_card.jpg', filtered_image)
    messagebox.showinfo("Success", "Filtered ID card saved as 'filtered_id_card.jpg'.")

def create_gui():
    global rotate_label, rotate_90_button, rotate_180_button, rotate_270_button
    global filter_label, grayscale_button, blur_button, edge_detection_button
    global crop_button

    root = tk.Tk()
    root.title("ID Card Scanner")

    label = tk.Label(root, text="Select an option:")
    label.pack()

    single_button = tk.Button(root, text="Capture Single Side", command=capture_single_side)
    single_button.pack()

    first_button = tk.Button(root, text="Capture First Side", command=capture_first_side)
    first_button.pack()

    second_button = tk.Button(root, text="Capture Second Side", command=capture_second_side)
    second_button.pack()

    rotate_label = tk.Label(root, text="Rotate:")
    rotate_90_button = tk.Button(root, text="90°", command=lambda: on_rotate_click(90))
    rotate_180_button = tk.Button(root, text="180°", command=lambda: on_rotate_click(180))
    rotate_270_button = tk.Button(root, text="270°", command=lambda: on_rotate_click(270))

    filter_label = tk.Label(root, text="Apply Filter:")
    grayscale_button = tk.Button(root, text="Grayscale", command=lambda: on_filter_click('grayscale'))
    blur_button = tk.Button(root, text="Blur", command=lambda: on_filter_click('blur'))
    edge_detection_button = tk.Button(root, text="Edge Detection", command=lambda: on_filter_click('edge'))

    crop_button = tk.Button(root, text="Crop Manually", command=crop_manually)

    root.mainloop()

def save_to_file_explorer(filename):
    filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", ".jpg"), ("All files", ".*")])
    if filepath:
        try:
            img = cv2.imread(filename)
            cv2.imwrite(filepath, img)
            messagebox.showinfo("Success", f"Saved as '{filepath}'")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

def main():
    create_gui()

if __name__ == "_main_":
    main()