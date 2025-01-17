from torch.utils.data import Dataset
from disent.data.groundtruth import XYSquaresData, GroundTruthData
from disent.dataset.groundtruth import GroundTruthDatasetPairs
from disent.transform import ToStandardisedTensor, FftBoxBlur

data: GroundTruthData = XYSquaresData(square_size=1, grid_size=2, num_squares=2)
dataset: Dataset = GroundTruthDatasetPairs(data, transform=ToStandardisedTensor(), augment=FftBoxBlur(radius=1, p=1.0))

for obs in dataset:
    # if augment is not None so the augmented 'x' exists in the observation
    (x0, x1), (x0_targ, x1_targ) = obs['x'], obs['x_targ']
    print(x0.dtype, x0.min(), x0.max(), x0.shape)
