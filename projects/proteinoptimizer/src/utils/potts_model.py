# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Potts models derived from direct coupling analysis (DCA).
Copied locally for Syn-3bfo protein landscape evaluation."""

from __future__ import annotations

import functools
from typing import Sequence

import numpy as np

# ---- Utility -------------------------------------------------------------

def onehot(labels, num_classes):
    """Convert integer encoded sequences to one-hot format."""
    if len(np.shape(labels)) == 3:  # already one-hot
        return labels
    if isinstance(labels, list):
        labels = np.asarray(labels)
    x = (labels[..., None] == np.arange(num_classes)[None])
    return x.astype(np.float32)

# ---- Internal helper functions (identical to original) -------------------

def _get_shifted_weights(weight_matrix: np.ndarray, wt_onehot_seq: np.ndarray, epi_offset: float = 0.0):
    modified_weights = np.copy(weight_matrix)
    offset_mat = np.einsum("in,jm->ijnm", wt_onehot_seq, wt_onehot_seq)
    for i in range(offset_mat.shape[0]):
        for m in range(offset_mat.shape[-1]):
            offset_mat[i, i, m, m] = 0.0
    modified_weights += -epi_offset * offset_mat
    return np.asarray(modified_weights)

def _get_dist_cutoff_weights(weight_matrix, distance_threshold):
    modified_weights = np.copy(weight_matrix)
    length = modified_weights.shape[0]
    for i in range(length):
        for j in range(length):
            if abs(i - j) < distance_threshold:
                modified_weights[i, j, :, :] = 0.0
    return np.asarray(modified_weights)

def _get_shifted_fields(field_vec, single_mut_offset, epi_offset, wt_onehot_seq):
    shifted_fields = np.copy(field_vec)
    single_mut_correction = single_mut_offset * wt_onehot_seq
    seq_len = wt_onehot_seq.shape[0]
    epi_correction = epi_offset * (seq_len - 1) * wt_onehot_seq
    shifted_fields += epi_correction + single_mut_correction
    return shifted_fields

def _slice_params_to_subsequence(field_vec, weight_matrix, start_idx, end_idx):
    sliced_field_vec = field_vec[start_idx:end_idx, :]
    idx_range = range(start_idx, end_idx)
    vocab_range = range(field_vec.shape[1])
    sliced_weight_matrix = weight_matrix[np.ix_(idx_range, idx_range, vocab_range, vocab_range)]
    return sliced_field_vec, sliced_weight_matrix

def is_valid_couplings(couplings_llaa):
    return np.allclose(couplings_llaa, couplings_llaa.transpose(1, 0, 3, 2))

# ---- Main PottsModel -----------------------------------------------------

class PottsModel:
    """Black-box objective based on (negative) Potts energy."""

    def __init__(
        self,
        weight_matrix: np.ndarray,
        field_vec: np.ndarray,
        wt_seq: Sequence[int],
        coupling_scale: float = 1.0,
        field_scale: float = 1.0,
        single_mut_offset: float = 0.0,
        epi_offset: float = 0.0,
        start_idx: int = 0,
        end_idx: int | None = None,
        distance_threshold_for_nearby_residues: int = 1,
        center_fitness_to_wildtype: bool = True,
    ) -> None:
        if not is_valid_couplings(weight_matrix):
            raise ValueError("Couplings tensor must be symmetric.")
        self._weight_matrix = weight_matrix
        self._field_vec = np.asarray(field_vec)
        self._vocab_size = self._field_vec.shape[1]
        self._start_idx = start_idx
        self._end_idx = end_idx if end_idx is not None else self._field_vec.shape[0]
        self._length = self._end_idx - self._start_idx

        self._field_vec, self._weight_matrix = _slice_params_to_subsequence(
            self._field_vec, self._weight_matrix, self._start_idx, self._end_idx
        )

        self._wt_seq = wt_seq[self._start_idx : self._end_idx]
        self._wt_onehot_seq = onehot([self._wt_seq], num_classes=self._vocab_size)[0]

        self._field_vec = _get_shifted_fields(self._field_vec, single_mut_offset, epi_offset, self._wt_onehot_seq)
        self._weight_matrix = _get_shifted_weights(self._weight_matrix, self._wt_onehot_seq, epi_offset)
        self._weight_matrix = _get_dist_cutoff_weights(self._weight_matrix, distance_threshold_for_nearby_residues)

        self._quad_deriv = np.einsum("ijkl,jl->ik", self._weight_matrix, self._wt_onehot_seq)
        self._coupling_scale = coupling_scale
        self._field_scale = field_scale
        self._center_fitness_to_wildtype = center_fitness_to_wildtype
        if center_fitness_to_wildtype:
            self._wildtype_fitness = -self._potts_energy(np.array([self.wildtype_sequence]))[0]

    # ------------------------------------------------------------------
    def evaluate(self, sequences):
        fitnesses = -self._potts_energy(sequences)
        if self._center_fitness_to_wildtype:
            fitnesses -= self._wildtype_fitness
        return fitnesses

    # Properties --------------------------------------------------------
    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def length(self):
        return self._length

    @property
    def wildtype_sequence(self):
        return self._wt_seq

    # Internal ----------------------------------------------------------
    def _potts_energy(self, sequences):
        if len(np.asarray(sequences).shape) == 1:
            sequences = np.reshape(sequences, (1, -1))
        onehot_seq = onehot(sequences, num_classes=self._vocab_size)
        linear_term = self._field_scale * np.einsum(
            "ij,bij->b", self._field_vec, onehot_seq, optimize="optimal"
        ) + (self._field_scale - self._coupling_scale) * np.einsum(
            "ij,bij->b", self._quad_deriv, onehot_seq, optimize="optimal"
        )
        quadratic_term = self._coupling_scale * 0.5 * np.einsum(
            "ijkl,bik,bjl->b", self._weight_matrix, onehot_seq, onehot_seq, optimize="optimal"
        )
        return linear_term + quadratic_term

# ---- Loader -------------------------------------------------------------

def load_from_mogwai_npz(filepath, **init_kwargs):
    with open(filepath, "rb") as f:
        state_dict = np.load(f)
        couplings = -1 * state_dict["weight"]
        bias = -1 * state_dict["bias"]
        wt_seq = state_dict["query_seq"]
    couplings = np.moveaxis(couplings, [0, 1, 2, 3], [0, 2, 1, 3])
    return PottsModel(weight_matrix=couplings, field_vec=bias, wt_seq=wt_seq, **init_kwargs) 