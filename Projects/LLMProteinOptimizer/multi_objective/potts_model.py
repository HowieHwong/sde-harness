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

"""Potts models derived from direct coupling analysis (DCA)."""

import functools
from typing import Sequence

import numpy as np

def onehot(labels, num_classes):
    """Convert integer encoded sequences to onehot format.

    Args:
      labels: 2D ndarray of integer-encoded sequences, num_seqs x seq_len
      num_classes: Number of classes

    Returns:
      3D onehot encoded array, num_seqs x seq_len x num_classes
    """
    if len(np.shape(labels)) == 3:  # if already one-hot, return labels
        return labels
    elif isinstance(labels, list):
        labels = np.asarray(labels)  # ndarray view
    x = (labels[Ellipsis, None] == np.arange(num_classes)[None])
    return x.astype(np.float32)


def _get_shifted_weights(weight_matrix: np.ndarray,
                         wt_onehot_seq: np.ndarray,
                         epi_offset: float=0.0):
    """Add correction for epistatic offset.

    Args:
      weight_matrix: 4D ndarray of couplings.
      wt_onehot_seq: One-hot encoded wildtype sequence.
      epi_offset: Shift of the mean of the pairwise epistasis distribution
        (F_{12}-F_{1}-F_{2}+F_{0}, computed relative to wildtype).

    Returns:
      ndarray view of copy of original weight matrix with shifted second order
      interactions
    """
    modified_weights = np.copy(weight_matrix)

    # Epistasis offset, quadratic term. Outer product of one-hot WT sequence.
    offset_mat = np.einsum('in,jm->ijnm', wt_onehot_seq, wt_onehot_seq)
    # remove diagonal of offset
    for i in range(offset_mat.shape[0]):
        for m in range(offset_mat.shape[-1]):
            offset_mat[i, i, m, m] = 0.0

    modified_weights += -epi_offset * offset_mat

    return np.asarray(modified_weights)


def _get_dist_cutoff_weights(weight_matrix,
                             distance_threshold):
    """Zeros nearby couplings given by `distance_threshold`.

    Args:
      weight_matrix: 4D ndarray of couplings.
      distance_threshold: Distance cutoff for zeroing. 0 gives no adjustment, 1
        gives 0 on the diagonal only, etc.

    Returns:
      ndarray view of copy of original weight matrix, with filtered near-diagonal
        elements.
    """
    modified_weights = np.copy(weight_matrix)
    length = modified_weights.shape[0]

    for i in range(length):
        for j in range(length):
            if abs(i - j) < distance_threshold:
                modified_weights[i, j, :, :] = 0.0

    return np.asarray(modified_weights)


def _get_shifted_fields(field_vec, single_mut_offset,
                        epi_offset, wt_onehot_seq):
    """Shifts fields to adjust single mutant effects and epistasis distributions.

    Args:
      field_vec: 2D ndarray of fields.
      single_mut_offset: Shift of single mutant fitness effects.
      epi_offset: Shift of epistasis distribution.
      wt_onehot_seq: One-hot encoded wildtype sequence.

    Returns:
      ndarray view of copy of original field vectors, with single mutation effect
        and epistasis shifts (relative to wildtype).
    """
    shifted_fields = np.copy(field_vec)

    single_mut_correction = single_mut_offset * wt_onehot_seq

    # epistasis corrections
    seq_len = wt_onehot_seq.shape[0]  # sequence length
    epi_correction = epi_offset * (seq_len - 1) * wt_onehot_seq

    shifted_fields += epi_correction + single_mut_correction

    return shifted_fields


def _slice_params_to_subsequence(field_vec,
                                 weight_matrix, start_idx,
                                 end_idx):
    """Crops a Potts model to use the position subset `start_idx`:`end_idx`.

    The `weight_matrix` is LxLxAxA. Subsetting the positions but maintaining
    the AxA interaction matrices returns a L'xL'xAxA tensor where
    L' = end_idx - start_idx.

    Args:
      field_vec: LxA vector.
      weight_matrix: LxLxAxA 4D tensor.
      start_idx: index to start cropping from.
      end_idx: index to stop cropping to.

    Returns:
      A tuple of (field, weight_matrix) arrays.
    """
    # TODO update the field term to account for lost pairwise terms
    sliced_field_vec = field_vec[start_idx:end_idx, :]
    idx_range = range(start_idx, end_idx)
    vocab_range = range(field_vec.shape[1])
    sliced_weight_matrix = weight_matrix[np.ix_(idx_range, idx_range, vocab_range,
                                                vocab_range)]
    return sliced_field_vec, sliced_weight_matrix


def is_valid_couplings(couplings_llaa):
    """Checks that the input coupling tensor is symmetric."""
    transposed_couplings_llaa = couplings_llaa.transpose(1, 0, 3, 2)
    is_symmetric = np.allclose(couplings_llaa, transposed_couplings_llaa)
    return is_symmetric


class PottsModel:
    """Black-box objective based on the negative energy of a Potts model.

      Model assumes no insert gap states.

      Tuning the Potts Model Objective:

      Includes parameters to independently control the mean of single mutant
      fitness effects as well as pairwise epistasis on double mutants
      (defined as F_{12}-F_{1}-F_{2}+F_{0}),
      with respect to wildtype.

      The single mutant fitness distribution is shifted
      by modifying the fields h with x_0, the one-hot representation of
      the wildtype:

              h' = h + single_mut_offset * x_0

      The mean of the pairwise epistasis distribution is shifted by modifying the
      couplings H and fields h by

              H' = H + epi_offset * x_0 (x_0)^T
              h' = h + epi_offset * L x_0

      where L is the length of the sequence.

      Afterwards, the distributions of the single mutant and pairwise epistasis
      distributions are independently scaled by field_scale and coupling_scale
      respectively by computing the energy E on a sequence x as

              E = coupling_scale * 0.5*(x)^T H x + field_scale h^T x
                    + (coupling_scale-field_scale) x_0^T H x

      There is also an option to filter interactions of nearby residues.
    """

    def __init__(self,
                 weight_matrix: np.ndarray,
                 field_vec: np.ndarray,
                 wt_seq: Sequence[int],
                 coupling_scale=1.0,
                 field_scale=1.0,
                 single_mut_offset=0.0,
                 epi_offset=0.0,
                 start_idx=0,
                 end_idx=None,
                 distance_threshold_for_nearby_residues=1,
                 center_fitness_to_wildtype=True):
        """Create an instance of this class.

        Args:
          weight_matrix: 4D ndarray, dimensions of L x L x A x A. Coupling matrix
            for Potts model.
          field_vec:  2D ndarray, L x A. Linear term in Potts model.
          wt_seq: Wildtype sequence. Integer-encoded list.
          coupling_scale: Scale factor for locally quadratic fitness changes (with
            respect to wildtype).
          field_scale:  Scale factor for single-site mutant fitness effects (with
            respect to wildtype).
          single_mut_offset: Shift of single mutant fitness change about wildtype.
          epi_offset: Shift of pairwise epistasis distribution (computed as
            F_{12}-F_{1}-F_{2}+F_{0} for mutants 1 and 2 on background 0) around
            wildtype sequence.
          start_idx: Model restricted to sub-sequence [start_idx:end_idx].
          end_idx: Model restricted to sub-sequence [start_idx:end_idx].
          distance_threshold_for_nearby_residues: Coordinates i,j in the sequence
            will be considered close to the diagonal if abs(i - j) < this. The
            couplings between these residues will be set to zero.
          center_fitness_to_wildtype: Whether to shift the output fitnesses such
            that the fitness of the wildtype is 0.
        """
        if not is_valid_couplings(weight_matrix):
            raise ValueError('Couplings tensor must be symmetric.')
        self._weight_matrix = weight_matrix
        self._field_vec = np.asarray(field_vec)
        self._vocab_size = self._field_vec.shape[1]

        self._start_idx = start_idx
        if end_idx is None:
            self._end_idx = self._field_vec.shape[0]
        else:
            self._end_idx = end_idx

        self._length = self._end_idx - self._start_idx

        # Take slices of couplings.
        self._field_vec, self._weight_matrix = _slice_params_to_subsequence(
            self._field_vec, self._weight_matrix, self._start_idx, self._end_idx)

        # Get WT sequence.
        self._wt_seq = wt_seq
        self._wt_seq = self._wt_seq[self._start_idx:self._end_idx]

        # One-hot representation for downstream calculations.
        self._wt_onehot_seq = onehot(
            [self._wt_seq], num_classes=self._vocab_size)[0]

        # Modify field terms for offsets
        self._field_vec = _get_shifted_fields(self._field_vec, single_mut_offset,
                                              epi_offset, self._wt_onehot_seq)
        self._weight_matrix = _get_shifted_weights(self._weight_matrix,
                                                   self._wt_onehot_seq, epi_offset)
        self._weight_matrix = _get_dist_cutoff_weights(
            self._weight_matrix, distance_threshold_for_nearby_residues)

        # First derivative of quadratic term at wildtype.
        # Result is seq_len x vocab_size (LxA).
        self._quad_deriv = np.einsum('ijkl,jl->ik', self._weight_matrix,
                                     self._wt_onehot_seq)

        self._coupling_scale = coupling_scale
        self._field_scale = field_scale
        self._center_fitness_to_wildtype = center_fitness_to_wildtype
        if center_fitness_to_wildtype:
            wt_array = np.array([self.wildtype_sequence, ])
            self._wildtype_fitness = -self._potts_energy(wt_array).item()

    def evaluate(self, sequences):
        fitnesses = -self._potts_energy(sequences)
        if self._center_fitness_to_wildtype:
            fitnesses -= self._wildtype_fitness
        return fitnesses

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def length(self):
        return self._length

    @property
    def wildtype_sequence(self):
        return self._wt_seq

    @property
    def weight_matrix(self):
        return self._weight_matrix

    @property
    def field_vec(self):
        return self._field_vec

    @property
    def coupling_scale(self):
        return self._coupling_scale

    @property
    def field_scale(self):
        return self._field_scale

    @property
    @functools.lru_cache()
    def epistasis_tensor(self):
        """Returns the epistasis tensor with respect to the wildtype sequence

        Recall that epistasis is given by:
            e_iAjB = H_iAjB - HiajB - HiAjb + Hiajb
        """
        H = self.weight_matrix
        L = H.shape[0]
        A = H.shape[2]
        epistasis_tensor = np.zeros_like(H)

        # TODO(nthomas) vectorize
        for i in range(L):
            for j in range(L):
                a = self.wildtype_sequence[i]
                b = self.wildtype_sequence[j]
                for alpha in range(A):
                    for beta in range(A):
                        epistasis_term = H[i, j, alpha, beta] - \
                            H[i, j, alpha, b] - H[i, j, a, beta] + H[i, j, a, b]
                        epistasis_tensor[i, j, alpha, beta] = epistasis_term
        return epistasis_tensor

    def _potts_energy(self, sequences):
        """Compute the Potts model energy."""
        if len(np.asarray(sequences).shape) == 1:  # single sequence
            sequences = np.reshape(sequences, (1, -1))
        # one-hot representation
        onehot_seq = onehot(sequences, num_classes=self._vocab_size)
        # (i, j, k, l, b) = (residue1, residue2, amino1, amino2, batch)
        linear_term = self._field_scale * np.einsum(
            'ij,bij->b', self._field_vec, onehot_seq, optimize='optimal') + (
                self._field_scale - self._coupling_scale) * np.einsum(
                    'ij,bij->b', self._quad_deriv, onehot_seq, optimize='optimal')
        quadratic_term = self._coupling_scale * 0.5 * np.einsum(
            'ijkl,bik,bjl->b',
            self._weight_matrix,
            onehot_seq,
            onehot_seq,
            optimize='optimal')

        return linear_term + quadratic_term


def load_from_mogwai_npz(filepath, **init_kwargs):
    """Load a landscape from a Potts Model state dict dumped from Mogwai.

    Args:
      filepath: A path to a .npz file with the following fields: ['weight',
        'bias', 'query_seq']. This file is assumed to be a saved state dict
        from the package mogwai https://github.com/nickbhat/mogwai.
      **init_kwargs: Kwargs passed to the PottsModel constructor.

    Returns:
      A PottsModel.
    """
    with open(filepath, 'rb') as f:
        state_dict = np.load(f)
        # Mogwai computes logits with a forward pass, so we need to invert
        # the couplings to get the expected energy computation
        couplings = -1 * state_dict['weight']
        bias = -1 * state_dict['bias']
        wt_seq = state_dict['query_seq']

    # Reshape the couplings from Mogwai. L, A, L, A -> L, L, A, A
    couplings = np.moveaxis(couplings, [0, 1, 2, 3], [0, 2, 1, 3])

    landscape = PottsModel(
        weight_matrix=couplings, field_vec=bias, wt_seq=wt_seq, **init_kwargs)
    return landscape
