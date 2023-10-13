import os
import csv
from scipy.io import loadmat
from itertools import combinations

def find_jpg_file_in_folder(path):
    folder_path = os.path.dirname(path)
    for file in os.listdir(folder_path):
        if file.lower().endswith('.jpg'):
            return file
    return None

# Define the root directory where your .mat files are located
root_directory = "./data/YorkUrbanDB"

# Create a list to store filenames that contain "vp_association"

vp_db=[]
# Recursively traverse the directory tree
for root, dirs, files in os.walk(root_directory):
    for filename in files:
        if filename.endswith(".mat"):
            file_path = os.path.join(root, filename)

            # Load the .mat file
            mat = loadmat(file_path, squeeze_me=True)

            # Check if "vp_association" is in the keys of the loaded mat
            if "vp_association" in mat:
                #print("processing...")
                #print(file_path)
                jpg_file = find_jpg_file_in_folder(file_path)
                original_array = mat['lines']
                fixedarray = []
                new_row=[]
                for i in range(0, len(original_array), 2):
                    element1 = original_array[i]
                    element2 = original_array[i + 1]
                    pair = [element1[0], element1[1], element2[0], element2[1]]
                    fixedarray.append(pair)

                # Sample value to append
                association = mat['vp_association']
                
                #print(association)
                # Iterate through fixedarray and add the value to each row
                #print(fixedarray)
                for i in range(len(fixedarray)):
                    fixedarray[i].append(association[i])

                data = fixedarray
                # Step 1: Group lines by their fifth value
                groups = {}
                for item in data:
                    group = int(item[4])  # Assuming the fifth value is an integer
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(item)

                # Step 2: Find the common intersection point for each group
                common_intersections = {}
                for group, lines in groups.items():
                    intersection_point = None
                    for line1, line2 in combinations(lines, 2):
                        
                        x1, y1, x2, y2 = line1[:4]
                        x3, y3, x4, y4 = line2[:4]
                        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                        if det != 0:
                            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
                            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

                            # Store the intersection point
                            intersection_point = (px, py)

                    if intersection_point:
                        common_intersections[group] = intersection_point

                # Print common intersection points for each group
                for group, intersection in common_intersections.items():
                    #print(f'Group {group}: Intersection point {intersection}')
                    new_row.append(intersection)
                #print("-----------------current vp")
                #print(new_row)
                value_vp=[new_row,jpg_file]
                vp_db.append(value_vp)

for i in range(10):
    print(vp_db[i])

filename = "data.csv"

with open(filename, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)

    for row in vp_db:
        csv_writer.writerow(row)