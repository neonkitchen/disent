import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader
#from disent.data.groundtruth import MultimodalXYSquaresData
from disent.data.groundtruth._multimodalxysquares import MultimodalXYSquaresData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.frameworks.vae import BetaVae
from disent.metrics import metric_dci, metric_mig
from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder
from disent.schedule import CyclicSchedule
from disent.transform import ToStandardisedTensor

data = MultimodalXYSquaresData()
dataset = GroundTruthDataset(data, transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)