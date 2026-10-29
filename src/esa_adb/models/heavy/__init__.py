import torch

# PyTorch's CPU thread parallelism isn't bit-stable run to run. That's noise
# below any metric's resolution for a single forward pass, but every heavy
# model here chains many partial_fit calls across walk-forward folds, and
# online gradient descent is sensitive to that kind of tiny perturbation. The
# same seed=42 Autoencoder run produced anomalies-only F0.5 values 2 to 4
# times apart depending only on thread count, with recall mostly unchanged
# but precision drifting. Pinning to one thread costs some speed but makes
# results reproducible, which matters more here than raw training speed.
torch.set_num_threads(1)

from esa_adb.models.heavy.anomaly_transformer_lite import AnomalyTransformerLite
from esa_adb.models.heavy.autoencoder import Autoencoder
from esa_adb.models.heavy.crtf_autoencoder import CRTFAutoencoder
from esa_adb.models.heavy.crtf_full import CRTFFull
from esa_adb.models.heavy.gdn import GDN
from esa_adb.models.heavy.loe_autoencoder import LOEAutoencoder
from esa_adb.models.heavy.lstm_forecaster import LSTMForecaster
from esa_adb.models.heavy.memstream import MemStream
from esa_adb.models.heavy.plda_autoencoder import PLDAAutoencoder
from esa_adb.models.heavy.roca_full import RoCAFull
from esa_adb.models.heavy.roca_lite import RoCALite

__all__ = ["AnomalyTransformerLite", "Autoencoder", "CRTFAutoencoder", "CRTFFull", "GDN", "LOEAutoencoder", "LSTMForecaster", "MemStream", "PLDAAutoencoder", "RoCAFull", "RoCALite"]
