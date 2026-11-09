# class ids used by the model (same as the training labels)

CLASS_NAMES = {
    1: "Tree cover",
    2: "Grassland",
    3: "Cropland",
    4: "Built-up",
    5: "Bare/sparse vegetation",
    6: "Permanent water",
}

CLASS_COLORS = {
    1: "#26734D",
    2: "#A8C66C",
    3: "#E6C75A",
    4: "#D95F59",
    5: "#B99B82",
    6: "#4A90C2",
}

CLASS_IDS = [1, 2, 3, 4, 5, 6]

# pixels with this label are skipped in training
IGNORE_INDEX = 255
