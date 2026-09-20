"""CPU implementation of Edge0's 4-bit affine expert matmul."""
import torch

def _as_bf16(x: torch.Tensor) -> torch.Tensor:
    """
    Checkpoint scales/biases are normally native BF16 tensors.

    Also accept floating-point tensors used by synthetic/unit tests,
    and uint16 tensors containing raw BF16 bit patterns.
    """
    if x.dtype == torch.bfloat16:
        return x

    if x.dtype == torch.uint16:
        return x.view(torch.bfloat16)

    if x.is_floating_point():
        return x

    raise TypeError(
        f"Expected BF16 or floating point tensor, got {x.dtype}"
    )


def unpack_int4_affine(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """
    Unpack 8 unsigned INT4 values from each uint32 word and
    dequantize using per-output-channel, per-group scale/bias.

    The final two dimensions are interpreted as:

        packed: [out_features, in_features // 8]
        scales: [out_features, in_features // group_size]
        biases: [out_features, in_features // group_size]

    Any leading dimensions are preserved.

    Output:

        [..., out_features, in_features]

    For the Edge0-8B checkpoint, a single expert has:

        packed: [512, 192]
        scales: [512, 24]
        biases: [512, 24]

    which produces:

        output: [512, 1536]

    The affine dequantization is:

        weight = (q - bias) * scale
    """

    if packed.ndim < 2:
        raise ValueError(
            f"packed must have at least 2 dimensions, "
            f"got {tuple(packed.shape)}"
        )

    if scales.ndim != packed.ndim or biases.ndim != packed.ndim:
        raise ValueError(
            "packed, scales, and biases must have the same "
            "number of dimensions"
        )

    if packed.dtype != torch.uint32:
        packed = packed.to(torch.uint32)

    scales = _as_bf16(scales)
    biases = _as_bf16(biases)

    values_per_word = 8

    # The final two packed dimensions are:
    #
    #   [..., out_features, packed_features]
    #
    out_features = packed.shape[-2]
    packed_features = packed.shape[-1]

    in_features = packed_features * values_per_word

    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by "
            f"group_size={group_size}"
        )

    expected_groups = in_features // group_size

    expected_scale_shape = (
        *packed.shape[:-1],
        expected_groups,
    )

    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scales shape {tuple(scales.shape)} does not match "
            f"expected {expected_scale_shape}"
        )

    if tuple(biases.shape) != expected_scale_shape:
        raise ValueError(
            f"biases shape {tuple(biases.shape)} does not match "
            f"expected {expected_scale_shape}"
        )

    # ---------------------------------------------------------------
    # Unpack the eight 4-bit values from every uint32 word.
    #
    # The packed word layout is:
    #
    #   word =
    #       q0
    #     | q1 << 4
    #     | q2 << 8
    #     | q3 << 12
    #     | q4 << 16
    #     | q5 << 20
    #     | q6 << 24
    #     | q7 << 28
    #
    # PyTorch CPU has limited uint32 operator support, so perform
    # the bit operations in int64.
    # ---------------------------------------------------------------

    packed_i64 = packed.to(torch.int64)

    shifts = (
        torch.arange(
            values_per_word,
            device=packed.device,
            dtype=torch.int64,
        )
        * 4
    )

    q = (packed_i64.unsqueeze(-1) >> shifts) & 0xF

    # q currently has shape:
    #
    #   [..., out_features, packed_features, 8]
    #
    # Collapse the packed dimension and the eight nibbles:
    #
    #   [..., out_features, in_features]
    q = q.reshape(
        *packed.shape[:-1],
        in_features,
    )

    # ---------------------------------------------------------------
    # Expand one scale/bias value across every group_size input
    # dimensions.
    #
    # Example for Edge0-8B:
    #
    #   scales: [512, 24]
    #
    # becomes:
    #
    #   [512, 1536]
    #
    # because:
    #
    #   24 groups * 64 values/group = 1536 values.
    # ---------------------------------------------------------------

    s = scales.to(torch.float32).repeat_interleave(
        group_size,
        dim=-1,
    )

    b = biases.to(torch.float32).repeat_interleave(
        group_size,
        dim=-1,
    )

    # The divisibility check above normally makes this unnecessary,
    # but retaining the trim makes the implementation robust if it
    # is later extended to non-divisible dimensions.
    s = s[..., :in_features]
    b = b[..., :in_features]

    # ---------------------------------------------------------------
    # Affine INT4 dequantization.
    # ---------------------------------------------------------------

    return (q.to(torch.float32) - b) * s


def expert_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """
    Apply a dequantized INT4 expert weight to an input tensor.

    Args:
        x:
            Input tensor whose final dimension is the input feature
            dimension.

        packed:
            Packed INT4 weight with shape:
                [out_features, in_features // 8]

        scales:
            Per-group scales with shape:
                [out_features, in_features // group_size]

        biases:
            Per-group affine biases with shape:
                [out_features, in_features // group_size]

    Returns:
        Tensor with final dimension equal to out_features.
    """

    weight = unpack_int4_affine(
        packed,
        scales,
        biases,
        group_size=group_size,
    )

    return torch.matmul(
        x.to(weight.dtype),
        weight.transpose(-1, -2),
    )
