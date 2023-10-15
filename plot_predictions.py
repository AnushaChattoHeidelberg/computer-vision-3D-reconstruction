import math
import cv2
import os
import numpy as np
import csv
import ast
def data(csv_file):
    data_paths = []  # Store image paths here
    labels = []
    
    with open(csv_file, newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Parse the string representation of the points using ast.literal_eval
            points = ast.literal_eval(row[0])

            # Extract the filename
            filename = row[1]
            filename, _ = os.path.splitext(filename)
            filename = "./data/YorkUrbanDB" + "/" + filename + "/" + filename + ".jpg"
            
            data_paths.append(filename)  # Store the image path
            
            # Calculate the centroid of the points and add the ground truth
            if points:
                x_coords, y_coords = zip(*points)
                centroid_x = sum(x_coords) / len(x_coords)
                centroid_y = sum(y_coords) / len(y_coords)
                centroid = [centroid_x, centroid_y]
                labels.append(centroid)
                
    return data_paths, labels

# Path to the folder where images are located
image_folder = "./data/YorkUrbanDB"

# Path to the folder containing .txt files
txt_folder = "./predictions2"

# Path to the folder to save modified images
predictions_folder = "./predictions2/plots"

# Load dataset and labels
data_paths, labels = data("data.csv")
original_labels=[]

#part 1

for txt_file in os.listdir(txt_folder):
    if txt_file.endswith(".txt"):
        txt_file_path = os.path.join(txt_folder, txt_file)

        # Extract image name from the .txt file name (without extension)
        image_name = os.path.splitext(txt_file)[0]

        # Path to the subfolder in the image folder
        subfolder_path = os.path.join(image_folder, image_name)

        # Check if the subfolder exists
        if os.path.exists(subfolder_path) and os is not None:
            # Load the image
            image_path = os.path.join(subfolder_path, f"{image_name}.jpg")
            image = cv2.imread(image_path)

            # Read x and y values from the .txt file
            x, y = None, None
            with open(txt_file_path, "r") as txt_file:
                lines = txt_file.readlines()
            for line in lines:
                if line.startswith("x:"):
                    x = float(line.split(":")[1].strip())
                elif line.startswith("y:"):
                    y = float(line.split(":")[1].strip())

            if x is not None and y is not None:
                # Calculate padding to ensure the point is within the image
                padding_top = max(0, int(-y))
                padding_bottom = max(0, int(y - image.shape[0]))
                padding_left = max(0, int(-x))
                padding_right = max(0, int(x - image.shape[1]))

                # Pad the image
                image_padded = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

                # Adjust the point's coordinates with the added padding
                x_adjusted = x + padding_left
                y_adjusted = y + padding_top

                # Draw a point on the padded image
                point_color = (0, 255, 0)  # Green color
                point_radius = 5
                point_thickness = -1  # Filled circle
                image_with_point = cv2.circle(image_padded.copy(), (int(x_adjusted), int(y_adjusted)), point_radius, point_color, point_thickness)

                # Save the modified image in the predictions folder
                output_image_path = os.path.join(predictions_folder, f"{image_name}_with_point.jpg")
                cv2.imwrite(output_image_path, image_with_point)
                original_labels.append([image_name,x,y])
            else:
                print(f"x and y values not found in {txt_file_path}")
        else:
            print(f"Subfolder {subfolder_path} not found for {txt_file_path}")

#part 2        

count=0
i=0

# Now, you can process images with labels as well, using the data_paths and labels obtained from a CSV file
for data_path, label in zip(data_paths, labels):
    image_name = os.path.splitext(os.path.basename(data_path))[0]
    image = cv2.imread(data_path)
    _,ground_truth_x,ground_truth_y = original_labels[i]

    if image is not None:
        x, y = label
        if math.sqrt((x - ground_truth_x)**2 + (y - ground_truth_y)**2) < 100:
            count=count+1
        padding_top = max(0, int(-y))
        padding_bottom = max(0, int(y - image.shape[0]))
        padding_left = max(0, int(-x))
        padding_right = max(0, int(x - image.shape[1]))

        image_padded = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        x_adjusted = x + padding_left
        y_adjusted = y + padding_top

        point_color = (0, 255, 0)  # Green color
        point_radius = 5
        point_thickness = -1  # Filled circle
        image_with_point = cv2.circle(image_padded.copy(), (int(x_adjusted), int(y_adjusted)), point_radius, point_color, point_thickness)

        # Save the modified image with a different name format
        output_image_path = os.path.join(predictions_folder, f"{image_name}_groundtruth.jpg")
        cv2.imwrite(output_image_path, image_with_point)
    else:
        print(f"Image not found at path: {data_path}")
    i=i+1

print("accuracy",count/len(labels))