import cv2
import numpy as np

#
# Function to preprocess the image
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# Function to segment ventricles
def segment_ventricles(image):
    # Apply thresholding to segment ventricles
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours of segmented regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area and find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask for the largest contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask

# Function to calculate volume from segmented mask
def calculate_volume(mask, pixel_spacing):
    # Calculate volume in cubic millimeters (mm^3)
    volume = np.sum(mask) * pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]
    return volume

# Function to process and calculate volume for each image
def process_and_calculate_volume(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Segment ventricles
    ventricle_mask = segment_ventricles(preprocessed_image)
    # Calculate pixel spacing (assuming isotropic voxels)
    pixel_spacing = (1.0, 1.0, 1.0)  # Example pixel spacing in millimeters (x, y, z)
    # Calculate volume
    volume = calculate_volume(ventricle_mask, pixel_spacing)
    return volume


def runEjectionFraciton(gender):
    # Paths to the images
    four_chamber_systole_path = 'test/outputs/main/4ch/end_systolic_frame.png'
    two_chamber_systole_path = 'test/outputs/main/2ch/end_systolic_frame.png'

    # Process and calculate volumes for each image
    four_chamber_systole_volume = process_and_calculate_volume(four_chamber_systole_path)
    two_chamber_systole_volume = process_and_calculate_volume(two_chamber_systole_path)


    # Paths to the images
    four_chamber_diastole_path = 'test/outputs/main/2ch/end_diastolic_frame.png'
    two_chamber_diastole_path = 'test/outputs/main/4ch/end_diastolic_frame.png'

    # Process and calculate volumes for each image
    four_chamber_diastole_volume = process_and_calculate_volume(four_chamber_diastole_path)
    two_chamber_diastole_volume = process_and_calculate_volume(two_chamber_diastole_path)

    # Print the calculated volumes
    EDV=(two_chamber_diastole_volume+four_chamber_diastole_volume)//2
    # Print the calculated volumes
    ESV=(two_chamber_systole_volume+four_chamber_systole_volume)//2

    EF=((EDV-ESV)/EDV)*100
    EF=round(EF,2)

    male_ef_ranges = {
        "Normal": (52, 72),
        "Mildly Reduced": (41, 51),
        "Moderately Reduced": (30, 40),
        "Severely Reduced": (0, 30)
    }

    female_ef_ranges = {
        "Normal": (54, 75),
        "Mildly Reduced": (41, 53),
        "Moderately Reduced": (30, 40),
        "Severely Reduced": (0, 30)
    }

    if gender == "male":
        if EF >= male_ef_ranges["Normal"][0] and EF <= male_ef_ranges["Normal"][1]:
            return ("Male EF",EF,"%"," gives heart condition is normal. ")
        elif EF >= male_ef_ranges["Mildly Reduced"][0] and EF <= male_ef_ranges["Mildly Reduced"][1]:
            return ("Male EF",EF,"%"," gives heart condition is Mildly abnormal. ")
        elif EF >= male_ef_ranges["Moderately Reduced"][0] and EF <= male_ef_ranges["Moderately Reduced"][1]:
            return ("Male EF",EF,"%"," gives heart condition is Moderately abnormal. ")
        else:
            return ("Male EF",EF,"%"," gives heart condition is Severe. ")

    elif gender == "female":
        if EF >= female_ef_ranges["Normal"][0] and EF <= female_ef_ranges["Normal"][1]: 
            return ("Female EF",EF,"%","gives heart condition is Normal")
        elif EF >= female_ef_ranges["Mildly Reduced"][0] and EF <= female_ef_ranges["Mildly Reduced"][1]:
            return ("Female EF",EF,"%","gives heart condition is Mildy Abnormal")
        elif EF >= female_ef_ranges["Moderately Reduced"][0] and EF <= female_ef_ranges["Moderately Reduced"][1]:
            return ("Female EF",EF,"%","gives heart condition is Moderately Abnormal")
        else:
            return ("Female EF",EF,"%","gives heart condition is Severe")
    else:
        return ("Invalid input. Please enter 'male' or 'female' as the gender.")
