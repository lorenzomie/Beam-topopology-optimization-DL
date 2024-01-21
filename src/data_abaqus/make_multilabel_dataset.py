"""
Script for generating multi-label dataset using Abaqus.

This script defines functions to generate a dataset by running an Abaqus script for each sample.
The dataset consists of arrays, each associated with displacement and weight values obtained from Abaqus simulations.
The script handles the creation of the dataset, runs Abaqus simulations, and saves the multi-label dataset.

Functions:
- generate_constrained_vector(length, max_value, max_unique_numbers=3)
- write_array_to_txt(array, array_file)
- create_dataset(dimension)
- run_abaqus_script(script_name)
- delete_files_by_name(base_name)
- delete_files_by_extension(extension)
- read_output_from_txt(output_file_path)
- create_multilabel_dataset(dataset, script_name, output_file_path, base_name)
- save_dataset(dataset, file_path)

Author: Lorenzo Miele
Date: 12/23
"""

import os
import glob
import time
import csv
import numpy as np
import subprocess

ARRAY_LENGTH = 100 # if you change this number you have to change the abaqus model file
DIM = 100
MATERIALS_NUMBER = 3

def generate_constrained_vector(length, max_value, max_unique_numbers=3):
    """
    Generate a random vector of specified length with values from 0 to max_value.

    Parameters:
    - length (int): Length of the generated vector.
    - max_value (int): Maximum value for the vector elements.
    - max_unique_numbers (int): Maximum number of unique values in the vector.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the random vector and its normalized version.
    """
    if max_unique_numbers > max_value:
        raise ValueError("The maximum number of unique numbers cannot exceed the maximum value.")

    unique_numbers = np.random.choice(max_value, size=min(max_unique_numbers, max_value), replace=False)
    random_vector = np.random.choice(np.arange(1, max_value + 1), size=length, replace=True)
    normalized_vector = random_vector / max_value
    return random_vector, normalized_vector

def write_array_to_txt(array, array_file):
    """
    Write a NumPy array to a text file.

    Parameters:
    - array (np.ndarray): NumPy array to be written to the file.
    - array_file (str): Path to the text file.

    Notes:
    - If the file already exists, it will be removed before writing the new array.
    """
    if os.path.exists(array_file):
        os.remove(array_file)
        print(f"File Removed: {array_file}")
    np.savetxt(array_file, array)
    
def create_dataset(dimension):
    """
    Create a dataset with random vectors.

    Parameters:
    - dimension (int): Number of vectors in the dataset.

    Returns:
    List[Tuple[np.ndarray, np.ndarray]]: List of tuples containing random vectors and their normalized versions.
    """
    dataset = []
    for i in range(dimension): 
        array, norm_array = generate_constrained_vector(ARRAY_LENGTH, MATERIALS_NUMBER)
        dataset.append((array, norm_array))
    return dataset

def run_abaqus_script(script_name):
    """
    Run an Abaqus script in the CAE without GUI environment.
    """
    try:
        # Construct the command to run the Abaqus script
        command = f'abaqus cae noGUI={script_name}'

        # Run the command in a subprocess
        subprocess.run(command, shell=True, check=True)
        print("Abaqus started with success")
    except subprocess.CalledProcessError as e:
        # Handle errors that may occur during script execution
        print(f"Error during Abaqus starting: {e}")

def delete_files_by_name(base_name):
    """
    Delete files with the specified base name and any extension.
    Example: delete_files_by_name("Beam") will delete Beam.odb, Beam.rec, Beam.com, etc.
    """
    try:
        # Get a list of file names with the specified base name and any extension
        file_names_to_delete = glob.glob(f"{base_name}.*")

        # Iterate over the list of file names
        for file_name in file_names_to_delete:
            # Check if the file exists
            if os.path.exists(file_name):
                # Remove the file
                os.remove(file_name)
                print(f"File Removed: {file_name}")
            else:
                print(f"File not found: {file_name}")

        print("Deletion complete")
    except Exception as e:
        # Handle any exceptions that may occur during file deletion
        print(f"Error during the file elimination: {e}")

def delete_files_by_extension(extension):
    """
    Delete files with the specified extension.
    Example: delete_files_by_extension("rec") will delete all files with the .rec extension.
    """
    try:
        # Get a list of file names with the specified extension
        file_names_to_delete = glob.glob(f"*.{extension}")

        # Iterate over the list of file names
        for file_name in file_names_to_delete:
            # Check if the file exists
            if os.path.exists(file_name):
                # Remove the file
                os.remove(file_name)
                print(f"File Removed: {file_name}")
            else:
                print(f"File not found: {file_name}")

        print("Deletion complete")
    except Exception as e:
        # Handle any exceptions that may occur during file deletion
        print(f"Error during the file elimination: {e}")

def read_output_from_txt(output_file_path):
    try:
        # Open the output file in read mode
        with open(output_file_path, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

        # Extract and convert max displacement from the first line
        max_displacement = float(lines[0].split(":")[1].strip())

        # Extract and convert total mass from the second line
        total_mass = float(lines[1].split(":")[1].strip())
        
        # Extract and convert frequencies from the third line
        frequencies = [float(f) for f in lines[2].split(":")[1].strip().split(",")]

        # Extract and convert the time output from the fourth line
        time_limit = float(lines[3].split(":")[1].strip())

        return max_displacement, total_mass, frequencies, time_limit
    
    except Exception as e:
        # Handle any exceptions that may occur during file reading
        print(f"Error during the file reading: {e}")
        
        return None, None, None, None

def create_multilabel_dataset(dataset, script_name, frequency_script_path,
        wave_script_path, output_file_path, base_names):
    # Initialize an empty list to store the multilabel dataset
    multilabel_dataset = []

    # Iterate over each entry in the dataset
    for i, my_tuple in enumerate(dataset):
        # Clean up existing files to prepare for the new Abaqus run
        # base_name is the model name
        for name in base_names:
            delete_files_by_name(name)
        delete_files_by_name("abaqus")
        delete_files_by_extension("rec")
        delete_files_by_extension("rec")

        # Write the current array to a text file
        write_array_to_txt(my_tuple[0], array_file_path)

        # Run the Abaqus script to perform the simulation
        run_abaqus_script(script_name)  # The script eliminates the output txt file before writing it
        time.sleep(1)
        run_abaqus_script(frequency_script_path)
        time.sleep(1)
        run_abaqus_script(wave_script_path)

        # Read the output values (displacement and weight) from the Abaqus simulation
        print(output_file_path)
        displacement, weight, frequencies, time_limit = read_output_from_txt(output_file_path)
        
        # Print diagnostic information for the current array
        print(f"displacement: {displacement}\nweight: {weight}\nfrequencies:{frequencies}", 
              f"\ntime output:{time_limit}\nArray {i}")
        print(f"array: {my_tuple[0]}")

        # Append the results to the multilabel dataset
        multilabel_dataset.append((my_tuple[1], displacement, weight, frequencies, time_limit))

    return multilabel_dataset

def save_dataset(dataset, file_path):
    try:
        # Check if the file exists before removing it
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File Removed: {file_path}")

        # Open the file in write mode using csv.writer
        with open(file_path, 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=';')

            # Write the header if the dataset is not empty
            if dataset:
                header = ['Vector', 'mass', 'displacement', 'frequencies', 'wave time reached']
                csv_writer.writerow(header)

                # Write each vector in the dataset to the file
                for current_vector, label1, label2, label3, label4 in dataset:
                    csv_writer.writerow([*current_vector, label1, label2, label3, label4])

            print(f"Dataset saved successfully: {file_path}")

    except Exception as e:
        # Handle any exceptions during file writing
        print(f"Error during file writing: {e}")

if __name__ == "__main__":
    # Get the folder containing the script
    script_folder = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths for temporary and training output folders
    output_folder_temporary_relative = os.path.join("..", "..", "data", "temporary")
    output_folder_relative = os.path.join("..", "..", "data", "training")

    # Define the names and paths for the Abaqus script, temporary files, and dataset
    abaqus_script_name = "wire_model.py"
    script_frequency_name = "wire_model_frequency.py"
    script_wave_name = "wire_model_implicit_wave.py"
    base_names = ["Beam", "Beam_frequency", "Beam_wave"]
    output_file_path = os.path.join(script_folder, output_folder_temporary_relative, "output_results.txt")
    array_file_path = os.path.join(script_folder, output_folder_temporary_relative, "array.txt")
    multilabel_dataset_path = os.path.join(script_folder, output_folder_relative, "multilabel_dataset.txt")
    abaqus_script_path = os.path.join(script_folder, abaqus_script_name)
    abaqus_frequency_script_path = os.path.join(script_folder, script_frequency_name)
    abaqus_wave_script_path = os.path.join(script_folder, script_wave_name)
    
    # Generate a dataset based on a specified dimension
    dataset = create_dataset(dimension=DIM)

    # Create a multilabel dataset by running the Abaqus script on each entry
    multilabel_dataset = create_multilabel_dataset(dataset, 
        abaqus_script_path, abaqus_frequency_script_path, 
        abaqus_wave_script_path, output_file_path, base_names)

    # Save the multilabel dataset to a file
    save_dataset(multilabel_dataset, multilabel_dataset_path)