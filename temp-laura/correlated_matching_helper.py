import math
import numpy as np
import pymatching
import sinter
import stim

class CorrelatedPyMatchingDecoder(sinter.Decoder):
    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path,
        dets_b8_in_path,
        obs_predictions_b8_out_path,
        tmp_dir,
    ):
        dem = stim.DetectorErrorModel.from_file(dem_path)
        matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=True
        )

        # Read bit-packed detectors: shape (num_shots, ceil(num_dets/8))
        num_det_bytes = math.ceil(num_dets / 8)
        dets_packed = np.fromfile(dets_b8_in_path, dtype=np.uint8)
        dets_packed = dets_packed.reshape(num_shots, num_det_bytes)

        # Unpack to bool array: shape (num_shots, num_dets)
        dets = np.unpackbits(dets_packed, axis=1, count=num_dets, bitorder="little")

        # Decode with correlations
        predictions = matching.decode_batch(dets, enable_correlations=True)
        # predictions shape: (num_shots, num_obs), dtype uint8 or bool

        # Pack predictions per shot: shape (num_shots, ceil(num_obs/8))
        num_obs_bytes = math.ceil(num_obs / 8)
        predictions_bool = predictions.astype(np.bool_)
        # Pad to multiple of 8 along obs axis if needed
        pad = num_obs_bytes * 8 - num_obs
        if pad > 0:
            predictions_bool = np.pad(predictions_bool, ((0, 0), (0, pad)))
        packed = np.packbits(predictions_bool, axis=1, bitorder="little")  # (num_shots, num_obs_bytes)

        packed.tofile(obs_predictions_b8_out_path)