import os
import re

list_a = list()
list_b = list()

def extract_numbers_from_filenames(folder_path):
    """
    Extracts numbers from the names of files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing the files.
    
    Returns:
        dict: A dictionary where the keys are filenames and the values are lists of extracted numbers.
    """
    # Regular expression to match numbers (including integers and decimals)
    number_pattern = re.compile(r'\d+')

    # Dictionary to store numbers extracted from each file name
    extracted_numbers = {}

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Extract numbers from the filename
        numbers = number_pattern.findall(filename)
        
        if numbers:
            # Convert the extracted numbers from strings to integers
            extracted_numbers[filename] = [int(num) for num in numbers]

    return extracted_numbers


# Example usage
folder_path = '../../data/raw_fault'

result = extract_numbers_from_filenames(folder_path)

# Output the results
if result:
    for filename, numbers in result.items():
        list_a.append(numbers[0])
        # print(f"File: {filename} | Extracted Numbers: {numbers}")
else:
    print("No numbers found in any file names.")


# Example usage
folder_path = '../../data/raw_seismic'

result = extract_numbers_from_filenames(folder_path)

# Output the results
if result:
    for filename, numbers in result.items():
        list_b.append(numbers[0])
        # print(f"File: {filename} | Extracted Numbers: {numbers}")
else:
    print("No numbers found in any file names.")

def find_intersection(list_a, list_b):
    """
    Finds intersecting numbers between two lists.
    
    Args:
        list_a (list): First list of numbers.
        list_b (list): Second list of numbers.
    
    Returns:
        list: A sorted list of intersecting numbers.
    """
    # Convert lists to sets and find the intersection
    intersection = set(list_a).intersection(set(list_b))
    
    # Convert the result back to a sorted list
    return sorted(intersection)

intersecting_numbers = find_intersection(list_a, list_b)

# Output the results
if intersecting_numbers:
    print(f"Found {len(intersecting_numbers)} intersecting numbers:")
    print(intersecting_numbers)
else:
    print("No intersecting numbers found.")
