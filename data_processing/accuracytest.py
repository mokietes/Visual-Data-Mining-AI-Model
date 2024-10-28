
from datasets import load_dataset

# Step 2: Define IoU calculation function
def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Box format: [x_min, y_min, x_max, y_max].
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# Step 3: Load dataset from Hugging Face
# Using the dataset agentsea/wave-ui-25k
dataset = load_dataset("agentsea/wave-ui-25k")

# Step 4: Define the keys for ground truth and predicted bounding boxes
# For this dataset, let's assume the boxes are stored in 'bbox' and 'pred_bbox'
# Update these based on the actual dataset structure if needed
ground_truth_key = 'bbox'  # Replace with actual key if needed
predicted_key = 'pred_bbox'  # Replace with actual key if needed

# Step 5: Set IoU threshold for accuracy
threshold = 0.5  # You can change this value (e.g., 0.75 for stricter accuracy)

# Step 6: Loop through the dataset and calculate IoU for each sample
accuracy_count = 0
total_samples = len(dataset['train'])  # Assuming we are using the 'train' split

for sample in dataset['train']:
    ground_truth_box = sample[ground_truth_key]
    predicted_box = sample[predicted_key]
    
    # Calculate IoU between the ground truth and predicted boxes
    iou = calculate_iou(ground_truth_box, predicted_box)
    
    # Check if the IoU meets the accuracy threshold
    if iou >= threshold:
        accuracy_count += 1

# Step 7: Calculate overall bounding box accuracy
accuracy = accuracy_count / total_samples
print(f'Bounding Box Accuracy: {accuracy * 100:.2f}%')

