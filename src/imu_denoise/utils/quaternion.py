"""Quaternion math utilities for IMU pose processing.

All quaternions use the scalar-first (Hamilton) convention: ``[w, x, y, z]``.
Functions accept single quaternions of shape ``(4,)`` or batches of shape ``(N, 4)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _ensure_batch(q: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
    """Promote a single quaternion to a batch of one. Returns (q, was_single)."""
    if q.ndim == 1:
        return q[np.newaxis, :], True
    return q, False


def quat_multiply(q1: NDArray[np.floating], q2: NDArray[np.floating]) -> NDArray[np.floating]:
    """Hamilton product of two quaternions (scalar-first convention).

    Args:
        q1: Quaternion(s) of shape ``(4,)`` or ``(N, 4)``.
        q2: Quaternion(s) of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        Product quaternion(s). Supports matching batch sizes, or a single
        quaternion multiplied against a batched input.
    """
    q1, single1 = _ensure_batch(q1)
    q2, single2 = _ensure_batch(q2)

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    result = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )

    if single1 and single2:
        return result[0]
    return result


def quat_conjugate(q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the conjugate of a quaternion (negate the vector part).

    Args:
        q: Quaternion(s) of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        Conjugate quaternion(s), same shape as input.
    """
    conj = np.array(q, dtype=q.dtype, copy=True)
    if conj.ndim == 1:
        conj[1:] *= -1
    else:
        conj[:, 1:] *= -1
    return conj


def quat_to_rotation_matrix(q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert unit quaternion(s) to 3x3 rotation matrix/matrices.

    Args:
        q: Quaternion(s) of shape ``(4,)`` or ``(N, 4)``.

    Returns:
        Rotation matrix of shape ``(3, 3)`` or ``(N, 3, 3)``.
    """
    q, single = _ensure_batch(q)

    # Normalize to unit quaternion
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = np.zeros((len(q), 3, 3), dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    if single:
        return R[0]
    return R


def quat_to_angular_velocity(
    q: NDArray[np.floating],
    timestamps: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute angular velocity in the body frame from a quaternion time series.

    Uses numerical differentiation (central differences where possible) and the
    relation ``omega = 2 * q_conj * dq/dt`` (vector part).

    Args:
        q: Quaternion time series of shape ``(N, 4)``.
        timestamps: Timestamps of shape ``(N,)`` in seconds.

    Returns:
        Angular velocity in rad/s of shape ``(N, 3)``.
    """
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternions of shape (N, 4), got {q.shape}")
    if timestamps.shape[0] != q.shape[0]:
        raise ValueError("timestamps length must match number of quaternions")

    n = q.shape[0]

    # Numerical derivative via central differences (forward/backward at edges)
    dq = np.zeros_like(q)
    dt = np.zeros(n, dtype=q.dtype)

    # Central differences for interior points
    dt[1:-1] = timestamps[2:] - timestamps[:-2]
    dq[1:-1] = q[2:] - q[:-2]

    # Forward difference for first point
    dt[0] = timestamps[1] - timestamps[0]
    dq[0] = q[1] - q[0]

    # Backward difference for last point
    dt[-1] = timestamps[-1] - timestamps[-2]
    dq[-1] = q[-1] - q[-2]

    # Avoid division by zero
    dt = np.maximum(dt, 1e-12)
    dq_dt = dq / dt[:, np.newaxis]

    # omega = 2 * conj(q) * dq/dt  (take vector part)
    q_conj = quat_conjugate(q)
    omega_quat = 2.0 * quat_multiply(q_conj, dq_dt)

    # Return only vector part [x, y, z]
    return omega_quat[:, 1:]


def rotate_vector(
    q: NDArray[np.floating],
    v: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Rotate vector(s) by quaternion(s) using the sandwich product ``q * v * q_conj``.

    Args:
        q: Quaternion(s) of shape ``(4,)`` or ``(N, 4)``.
        v: Vector(s) of shape ``(3,)`` or ``(N, 3)``.

    Returns:
        Rotated vector(s), same shape as ``v``.
    """
    single_q = q.ndim == 1
    single_v = v.ndim == 1

    if single_q:
        q = q[np.newaxis, :]
    if single_v:
        v = v[np.newaxis, :]

    # Embed vectors as pure quaternions [0, vx, vy, vz]
    v_quat = np.zeros((v.shape[0], 4), dtype=v.dtype)
    v_quat[:, 1:] = v

    q_conj = quat_conjugate(q)
    rotated = quat_multiply(quat_multiply(q, v_quat), q_conj)

    result = rotated[:, 1:]
    if single_q and single_v:
        return result[0]
    return result
