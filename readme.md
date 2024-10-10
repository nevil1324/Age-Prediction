# SMAI - Assignment4

### Method used

- Done data-augmentation
  (added transformation in image).

```
Resize
RandomHorizontalFlip
RandomRotation
ColorJitter
GaussianBlur

```

- Apart from this, I have divided train-data into train and validation set.
- The training-validation split has been conducted while preserving the age distribution present in the original dataset.

### Model Used

- In the final solution, I've used pre-trained ResNet18 model checkpoints and trained it further on our dataset.

- While backprop loss , i have done masking on loss.
  `if abs(predicted - original) <= k then loss = 0`

```
First 30 Epochs => k = 2
Next 15 Epochs  => k = 1
Next 12 Epochs  => k = 0
Total = 57 Epochs Training
```

### Files

- 2023202028_A4.py -> python script for the age prediction.
- 2023202028_A4.csv -> output file.
