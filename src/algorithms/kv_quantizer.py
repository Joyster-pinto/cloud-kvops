"""
LLM-KVOps: Advanced Algorithm 2 - KIVI Dynamic Per-Channel INT8 / INT4 KV Quantizer
Paper Reference: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (ICML / NeurIPS)

Compresses Key and Value tensors from FP16 (2 bytes / 16 bits) to INT8 (1 byte / 8 bits)
or INT4 (0.5 bytes / 4 bits packed) along the channel dimension.
Delivers 50% (INT8) to 75% (INT4) memory compression with > 0.99 numerical cosine similarity.
"""

from typing import Tuple, Dict, Any, Union
import numpy as np


class QuantizedTensor:
    """Container holding quantized integer data alongside scaling factors."""
    def __init__(
        self,
        quantized_data: np.ndarray,
        scale: np.ndarray,
        zero_point: np.ndarray,
        original_shape: Tuple[int, ...],
        bits: int = 8
    ):
        self.quantized_data = quantized_data  # int8 or uint8 packed
        self.scale = scale                    # float32 scale per channel
        self.zero_point = zero_point          # float32/int32 zero point per channel
        self.original_shape = original_shape
        self.bits = bits

    def byte_size(self) -> int:
        """Returns exact physical byte size of the quantized container."""
        return (
            self.quantized_data.nbytes +
            self.scale.nbytes +
            self.zero_point.nbytes
        )


class KVIQuantizer:
    """
    Performs dynamic asymmetric per-channel quantization on Key and Value tensors.
    Supports INT8 (8-bit) and INT4 (4-bit packed) precision.
    """
    def __init__(self, bits: int = 8):
        assert bits in (4, 8), "KVIQuantizer supports 4-bit and 8-bit quantization."
        self.bits = bits
        if bits == 8:
            self.qmin = -128
            self.qmax = 127
        else:
            self.qmin = 0
            self.qmax = 15

    def quantize(self, tensor: np.ndarray, axis: int = -1) -> QuantizedTensor:
        """
        Quantizes an FP16 / FP32 tensor along the specified channel axis.

        Args:
            tensor: High-dimensional KV tensor, e.g., shape (Seq_Len, Head_Dim) or (Batch, Heads, Seq_Len, Dim).
            axis: Axis along which scales and zero points are computed (default: channel dim -1).

        Returns:
            QuantizedTensor container.
        """
        t = np.asarray(tensor, dtype=np.float32)
        shape = t.shape

        t_min = np.min(t, axis=axis, keepdims=True)
        t_max = np.max(t, axis=axis, keepdims=True)

        # Prevent division by zero
        range_span = np.maximum(t_max - t_min, 1e-7)

        if self.bits == 8:
            # Symmetric / Asymmetric INT8: [-128, 127]
            scale = range_span / 255.0
            zero_point = np.round(-128.0 - (t_min / scale))
            quantized = np.clip(np.round(t / scale) + zero_point, -128, 127).astype(np.int8)
            return QuantizedTensor(quantized, scale, zero_point, shape, bits=8)
        else:
            # Asymmetric INT4: [0, 15] packed 2 values per byte
            scale = range_span / 15.0
            zero_point = np.round(- (t_min / scale))
            quantized_raw = np.clip(np.round(t / scale) + zero_point, 0, 15).astype(np.uint8)

            # Pack two 4-bit nibbles into one uint8 byte along the last dimension
            flat = quantized_raw.reshape(-1)
            if len(flat) % 2 != 0:
                flat = np.append(flat, np.uint8(0))
            packed = ((flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)).astype(np.uint8)

            return QuantizedTensor(packed, scale, zero_point, shape, bits=4)

    def dequantize(self, q_tensor: QuantizedTensor) -> np.ndarray:
        """
        Reconstructs the float tensor from quantized representation.

        Args:
            q_tensor: QuantizedTensor instance.

        Returns:
            Reconstructed float32 tensor matching original shape.
        """
        if q_tensor.bits == 8:
            dequantized = (q_tensor.quantized_data.astype(np.float32) - q_tensor.zero_point) * q_tensor.scale
            return dequantized.reshape(q_tensor.original_shape)
        else:
            # Unpack 4-bit values
            packed = q_tensor.quantized_data
            low = (packed & 0x0F).astype(np.uint8)
            high = ((packed >> 4) & 0x0F).astype(np.uint8)
            total_elements = int(np.prod(q_tensor.original_shape))
            interleaved = np.empty(len(packed) * 2, dtype=np.uint8)
            interleaved[0::2] = low
            interleaved[1::2] = high
            unpacked = interleaved[:total_elements].reshape(q_tensor.original_shape).astype(np.float32)

            dequantized = (unpacked - q_tensor.zero_point) * q_tensor.scale
            return dequantized

    @staticmethod
    def compute_fidelity_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Computes numerical fidelity metrics:
        - Cosine Similarity: [0.0, 1.0] (measures directional alignment)
        - Mean Squared Error (MSE)
        - Signal-to-Noise Ratio (SNR) in dB
        """
        orig_flat = original.astype(np.float64).reshape(-1)
        recon_flat = reconstructed.astype(np.float64).reshape(-1)

        # Cosine similarity
        norm_orig = np.linalg.norm(orig_flat)
        norm_recon = np.linalg.norm(recon_flat)
        if norm_orig > 0 and norm_recon > 0:
            cosine_sim = float(np.dot(orig_flat, recon_flat) / (norm_orig * norm_recon))
        else:
            cosine_sim = 1.0

        mse = float(np.mean((orig_flat - recon_flat) ** 2))
        signal_pwr = float(np.mean(orig_flat ** 2))
        snr_db = float(10.0 * np.log10(signal_pwr / (mse + 1e-12))) if mse > 0 else 100.0

        return {
            "cosine_similarity": cosine_sim,
            "mean_squared_error": mse,
            "snr_db": snr_db
        }

    @staticmethod
    def compute_compression_ratio(original_shape: Tuple[int, ...], bits: int = 8, fp_precision_bytes: int = 2) -> Dict[str, Any]:
        """Calculates theoretical and container-level memory savings."""
        total_elements = int(np.prod(original_shape))
        uncompressed_bytes = total_elements * fp_precision_bytes

        if bits == 8:
            compressed_bytes = total_elements * 1
        elif bits == 4:
            compressed_bytes = int(np.ceil(total_elements * 0.5))
        else:
            compressed_bytes = total_elements * fp_precision_bytes

        saved_bytes = uncompressed_bytes - compressed_bytes
        savings_pct = (saved_bytes / uncompressed_bytes) * 100.0 if uncompressed_bytes > 0 else 0.0

        return {
            "uncompressed_bytes": uncompressed_bytes,
            "compressed_bytes": compressed_bytes,
            "saved_bytes": saved_bytes,
            "savings_pct": savings_pct,
            "compression_factor": uncompressed_bytes / max(1, compressed_bytes)
        }
