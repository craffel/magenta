# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MusicVAE data library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import copy
import functools
import itertools
import random

# internal imports
import numpy as np
import tensorflow as tf

import magenta.music as mm
from magenta.music import drums_encoder_decoder
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2
from tensorflow.python.util import nest

PIANO_MIN_MIDI_PITCH = 21
PIANO_MAX_MIDI_PITCH = 108
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIDI_PITCHES = 128

MAX_INSTRUMENT_NUMBER = 127

MEL_PROGRAMS = range(0, 32)  # piano, chromatic percussion, organ, guitar
BASS_PROGRAMS = range(32, 40)
ELECTRIC_BASS_PROGRAM = 33

REDUCED_DRUM_PITCH_CLASSES = drums_encoder_decoder.DEFAULT_DRUM_TYPE_PITCHES
FULL_DRUM_PITCH_CLASSES = [  # 61 classes
    [p] for c in drums_encoder_decoder.DEFAULT_DRUM_TYPE_PITCHES for p in c]


def _maybe_pad_seqs(seqs, dtype):
  """Pads sequences to match the longest and returns as a numpy array."""
  if not len(seqs):  # pylint:disable=g-explicit-length-test
    return np.zeros((0, 0, 0), dtype)
  lengths = [len(s) for s in seqs]
  if len(set(lengths)) == 1:
    return np.array(seqs, dtype)
  else:
    length = max(lengths)
    return (np.array([np.pad(s, [(0, length - len(s)), (0, 0)], mode='constant')
                      for s in seqs], dtype))


def _extract_instrument(note_sequence, instrument):
  extracted_ns = copy.copy(note_sequence)
  del extracted_ns.notes[:]
  extracted_ns.notes.extend(
      n for n in note_sequence.notes if n.instrument == instrument)
  return extracted_ns


def np_onehot(indices, depth, dtype=np.bool):
  """Converts 1D array of indices to a one-hot 2D array with given depth."""
  onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
  onehot_seq[np.arange(len(indices)), indices] = 1.0
  return onehot_seq


class NoteSequenceAugmenter(object):
  """Class for augmenting NoteSequences.

  Args:
    transpose_range: A tuple containing the inclusive, integer range of
        transpose amounts to sample from. If None, no transposition is applied.
    stretch_range: A tuple containing the inclusive, float range of stretch
        amounts to sample from.
  Returns:
    The augmented NoteSequence.
  """

  def __init__(self, transpose_range=None, stretch_range=None):
    self._transpose_range = transpose_range
    self._stretch_range = stretch_range

  def augment(self, note_sequence):
    """Python implementation that augments the NoteSequence."""
    trans_amt = (random.randint(*self._transpose_range)
                 if self._transpose_range else 0)
    stretch_factor = (random.uniform(*self._stretch_range)
                      if self._stretch_range else 1.0)
    augmented_ns = copy.deepcopy(note_sequence)
    del augmented_ns.notes[:]
    for note in note_sequence.notes:
      aug_pitch = note.pitch + trans_amt
      if MIN_MIDI_PITCH <= aug_pitch <= MAX_MIDI_PITCH:
        augmented_ns.notes.add().CopyFrom(note)
        augmented_ns.notes[-1].pitch = aug_pitch

    augmented_ns = sequences_lib.stretch_note_sequence(
        augmented_ns, stretch_factor)
    return augmented_ns

  def tf_augment(self, note_sequence_scalar):
    """TF op that augments the NoteSequence."""
    def _augment_str(note_sequence_str):
      note_sequence = music_pb2.NoteSequence.FromString(note_sequence_str)
      augmented_ns = self.augment(note_sequence)
      return [augmented_ns.SerializeToString()]

    augmented_note_sequence_scalar = tf.py_func(
        _augment_str,
        [note_sequence_scalar],
        tf.string,
        name='augment')
    augmented_note_sequence_scalar.set_shape(())
    return augmented_note_sequence_scalar


class BaseConverter(object):
  """Base class for data converters between items and tensors.

  Inheriting classes must implement the following abstract methods:
    -`_to_tensors`
    -`_to_items`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, input_depth, input_dtype, output_depth, output_dtype,
               end_token, max_tensors_per_item=None,
               str_to_item_fn=lambda s: s, length_shape=()):
    """Initializes BaseConverter.

    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      end_token: Optional end token.
      max_tensors_per_item: The maximum number of outputs to return for each
        input.
      str_to_item_fn: Callable to convert raw string input into an item for
        conversion.
      length_shape: Shape of length returned by `to_tensor`.
    """
    self._input_depth = input_depth
    self._input_dtype = input_dtype
    self._output_depth = output_depth
    self._output_dtype = output_dtype
    self._end_token = end_token
    self._max_tensors_per_input = max_tensors_per_item
    self._str_to_item_fn = str_to_item_fn
    self._is_training = False
    self._length_shape = length_shape

  @property
  def is_training(self):
    return self._is_training

  @property
  def str_to_item_fn(self):
    return self._str_to_item_fn

  @is_training.setter
  def is_training(self, value):
    self._is_training = value

  @property
  def max_tensors_per_item(self):
    return self._max_tensors_per_input

  @max_tensors_per_item.setter
  def max_tensors_per_item(self, value):
    self._max_tensors_per_input = value

  @property
  def end_token(self):
    """End token, or None."""
    return self._end_token

  @property
  def input_depth(self):
    """Dimension of inputs (to encoder) at each timestep of the sequence."""
    return self._input_depth

  @property
  def input_dtype(self):
    """DType of inputs (to encoder)."""
    return self._input_dtype

  @property
  def output_depth(self):
    """Dimension of outputs (from decoder) at each timestep of the sequence."""
    return self._output_depth

  @property
  def output_dtype(self):
    """DType of outputs (from decoder)."""
    return self._output_dtype

  @property
  def length_shape(self):
    """Shape of length returned by `to_tensor`."""
    return self._length_shape

  @abc.abstractmethod
  def _to_tensors(self, item):
    """Implementation that converts item into list of tensors.

    Args:
     item: Item to convert.

    Returns:
      input_tensors: Tensors to feed to the encoder.
      output_tensors: Tensors to feed to the decoder.
    """
    pass

  @abc.abstractmethod
  def _to_items(self, samples):
    """Implementation that decodes model samples into list of items."""
    pass

  def _maybe_sample_outputs(self, outputs):
    """If should limit outputs, returns up to limit (randomly if training)."""
    if (not self.max_tensors_per_item or
        len(outputs) <= self.max_tensors_per_item):
      return outputs
    if self.is_training:
      indices = set(np.random.choice(
          len(outputs), size=self.max_tensors_per_item, replace=False))
      return [outputs[i] for i in indices]
    else:
      return outputs[:self.max_tensors_per_item]

  def to_tensors(self, item):
    """Python method that converts `item` into list of tensors."""
    inputs, outputs = self._to_tensors(item)
    lengths = [len(i) for i in inputs]
    sampled_results = self._maybe_sample_outputs(zip(inputs, outputs, lengths))
    return tuple(zip(*sampled_results)) if sampled_results else ([], [], [])

  def to_items(self, samples):
    """Python method that decodes samples into list of items."""
    return self._to_items(samples)

  def tf_to_tensors(self, item_scalar):
    """TensorFlow op that converts item into output tensors.

    Sequences will be padded to match the length of the longest.

    Args:
      item_scalar: A scalar of type tf.String containing the raw item to be
        converted to tensors.

    Returns:
      outputs: A Tensor, shaped [num encoded seqs, max(lengths), output_depth],
        containing the padded output encodings resulting from the input.
      lengths: A tf.int32 Tensor, shaped [num encoded seqs], containing the
        unpadded lengths of the tensor sequences resulting from the input.
    """
    def _convert_and_pad(item_str):
      item = self.str_to_item_fn(item_str)
      inputs, outputs, lengths = self.to_tensors(item)
      inputs = _maybe_pad_seqs(inputs, self.input_dtype)
      outputs = _maybe_pad_seqs(outputs, self.output_dtype)
      return inputs, outputs, np.array(lengths, np.int32)
    inputs, outputs, lengths = tf.py_func(
        _convert_and_pad,
        [item_scalar],
        [self.input_dtype, self.output_dtype, tf.int32],
        name='convert_and_pad')
    inputs.set_shape([None, None, self.input_depth])
    outputs.set_shape([None, None, self.output_depth])
    lengths.set_shape([None] + list(self.length_shape))
    return inputs, outputs, lengths


class BaseNoteSequenceConverter(BaseConverter):
  """Base class for NoteSequence data converters.

  Inheriting classes must implement the following abstract methods:
    -`_to_tensors`
    -`_to_notesequences`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, input_depth, input_dtype, output_depth, output_dtype,
               end_token, presplit_on_time_changes=True,
               max_tensors_per_notesequence=None):
    """Initializes BaseNoteSequenceConverter.

    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      end_token: Optional end token.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
        before converting.
      max_tensors_per_notesequence: The maximum number of outputs to return
        for each NoteSequence.
    """
    super(BaseNoteSequenceConverter, self).__init__(
        input_depth, input_dtype, output_depth, output_dtype, end_token,
        max_tensors_per_item=max_tensors_per_notesequence,
        str_to_item_fn=music_pb2.NoteSequence.FromString)

    self._presplit_on_time_changes = presplit_on_time_changes

  @property
  def max_tensors_per_notesequence(self):
    return self.max_tensors_per_item

  @max_tensors_per_notesequence.setter
  def max_tensors_per_notesequence(self, value):
    self.max_tensors_per_item = value

  @abc.abstractmethod
  def _to_notesequences(self, samples):
    """Implementation that decodes model samples into list of NoteSequences."""
    return

  def to_notesequences(self, samples):
    """Python method that decodes samples into list of NoteSequences."""
    return self._to_items(samples)

  def to_tensors(self, note_sequence):
    """Python method that converts `note_sequence` into list of tensors."""
    if self._presplit_on_time_changes:
      note_sequences = sequences_lib.split_note_sequence_on_time_changes(
          note_sequence)
    else:
      note_sequences = [note_sequence]
    results = []
    for ns in note_sequences:
      inputs, outputs = self._to_tensors(ns)
      if inputs:
        lengths = [len(i) for i in inputs]
        results.extend(zip(inputs, outputs, lengths))
    sampled_results = self._maybe_sample_outputs(results)
    return tuple(zip(*sampled_results)) if sampled_results else ([], [], [])

  def _to_items(self, samples):
    """Python method that decodes samples into list of NoteSequences."""
    return self._to_notesequences(samples)


class LegacyEventListOneHotConverter(BaseNoteSequenceConverter):
  """Converts NoteSequences using legacy OneHotEncoding framework.

  Quantizes the sequences, extracts event lists in the requested size range,
  uniquifies, and converts to encoding. Uses the OneHotEncoding's
  output encoding for both the input and output.

  Args:
    event_list_fn: A function that returns a new EventSequence.
    event_extractor_fn: A function for extracing events into EventSequences. The
      sole input should be the quantized NoteSequence.
    legacy_encoder_decoder: An instantiated OneHotEncoding object to use.
    add_end_token: Whether or not to add an end token. Recommended to be False
      for fixed-length outputs.
    slice_bars: Optional size of window to slide over raw event lists after
      extraction.
    steps_per_quarter: The number of quantization steps per quarter note.
      Mututally exclusive with `steps_per_second`.
    steps_per_second: The number of quantization steps per second.
      Mututally exclusive with `steps_per_quarter`.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    presplit_on_time_changes: Whether to split NoteSequence on time changes
     before converting.
  """

  def __init__(self, event_list_fn, event_extractor_fn,
               legacy_encoder_decoder, add_end_token=False, slice_bars=None,
               slice_steps=None, steps_per_quarter=None, steps_per_second=None,
               quarters_per_bar=4, pad_to_total_time=False,
               max_tensors_per_notesequence=None,
               presplit_on_time_changes=True):
    if (steps_per_quarter, steps_per_second).count(None) != 1:
      raise ValueError(
          'Exactly one of `steps_per_quarter` and `steps_per_second` should be '
          'provided.')
    if (slice_bars, slice_steps).count(None) == 0:
      raise ValueError(
          'At most one of `slice_bars` and `slice_steps` should be provided.')
    self._event_list_fn = event_list_fn
    self._event_extractor_fn = event_extractor_fn
    self._legacy_encoder_decoder = legacy_encoder_decoder
    self._steps_per_quarter = steps_per_quarter
    if steps_per_quarter:
      self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._steps_per_second = steps_per_second
    if slice_bars:
      self._slice_steps = self._steps_per_bar * slice_bars
    else:
      self._slice_steps = slice_steps
    self._pad_to_total_time = pad_to_total_time

    depth = legacy_encoder_decoder.num_classes + add_end_token
    super(LegacyEventListOneHotConverter, self).__init__(
        input_depth=depth,
        input_dtype=np.bool,
        output_depth=depth,
        output_dtype=np.bool,
        end_token=legacy_encoder_decoder.num_classes if add_end_token else None,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors(self, note_sequence):
    """Converts NoteSequence to unique, one-hot tensor sequences."""
    try:
      if self._steps_per_quarter:
        quantized_sequence = mm.quantize_note_sequence(
            note_sequence, self._steps_per_quarter)
        if (mm.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
            self._steps_per_bar):
          return [], []
      else:
        quantized_sequence = mm.quantize_note_sequence_absolute(
            note_sequence, self._steps_per_second)
    except (mm.BadTimeSignatureException, mm.NonIntegerStepsPerBarException,
            mm.NegativeTimeException) as e:
      return [], []

    event_lists, unused_stats = self._event_extractor_fn(quantized_sequence)
    if self._pad_to_total_time:
      for e in event_lists:
        e.set_length(len(e) + e.start_step, from_left=True)
        e.set_length(quantized_sequence.total_quantized_steps)
    if self._slice_steps:
      sliced_event_tuples = []
      for l in event_lists:
        for i in range(self._slice_steps, len(l) + 1, self._steps_per_bar):
          sliced_event_tuples.append(tuple(l[i - self._slice_steps: i]))
    else:
      sliced_event_tuples = [tuple(l) for l in event_lists]

    # TODO(adarob): Consider handling the fact that different event lists can
    # be mapped to identical tensors by the encoder_decoder (e.g., Drums).

    unique_event_tuples = list(set(sliced_event_tuples))
    unique_event_tuples = self._maybe_sample_outputs(unique_event_tuples)

    seqs = []
    for t in unique_event_tuples:
      seqs.append(np_onehot(
          [self._legacy_encoder_decoder.encode_event(e) for e in t] +
          ([] if self.end_token is None else [self.end_token]),
          self.output_depth, self.output_dtype))

    return seqs, seqs

  def _to_notesequences(self, samples):
    output_sequences = []
    for sample in samples:
      s = np.argmax(sample, axis=-1)
      if self.end_token is not None and self.end_token in s.tolist():
        s = s[:s.tolist().index(self.end_token)]
      event_list = self._event_list_fn()
      for e in s:
        assert e != self.end_token
        event_list.append(self._legacy_encoder_decoder.decode_event(e))
      output_sequences.append(event_list.to_sequence(velocity=80))
    return output_sequences


class OneHotMelodyConverter(LegacyEventListOneHotConverter):
  """Converter for legacy MelodyOneHotEncoding.

  Args:
    min_pitch: The minimum pitch to model. Those below this value will be
      ignored.
    max_pitch: The maximum pitch to model. Those above this value will be
      ignored.
    valid_programs: Optional set of program numbers to allow.
    skip_polyphony: Whether to skip polyphonic instruments. If False, the
      highest pitch will be taken in polyphonic sections.
    max_bars: Optional maximum number of bars per extracted melody, before
      slicing.
    slice_bars: Optional size of window to slide over raw Melodies after
      extraction.
    gap_bars: If this many bars or more of non-events follow a note event, the
       melody is ended. Disabled when set to 0 or None.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    add_end_token: Whether to add an end token at the end of each sequence.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
  """

  def __init__(self, min_pitch=PIANO_MIN_MIDI_PITCH,
               max_pitch=PIANO_MAX_MIDI_PITCH, valid_programs=None,
               skip_polyphony=False, max_bars=None, slice_bars=None,
               gap_bars=1.0, steps_per_quarter=4, quarters_per_bar=4,
               add_end_token=False, pad_to_total_time=False,
               max_tensors_per_notesequence=5, presplit_on_time_changes=True):
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch
    self._valid_programs = valid_programs
    steps_per_bar = steps_per_quarter * quarters_per_bar
    max_steps_truncate = steps_per_bar * max_bars if max_bars else None

    def melody_fn():
      return mm.Melody(
          steps_per_bar=steps_per_bar, steps_per_quarter=steps_per_quarter)
    melody_extractor_fn = functools.partial(
        mm.extract_melodies,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=max_steps_truncate,
        min_unique_pitches=1,
        ignore_polyphonic_notes=not skip_polyphony,
        pad_end=True)
    super(OneHotMelodyConverter, self).__init__(
        melody_fn,
        melody_extractor_fn,
        mm.MelodyOneHotEncoding(min_pitch, max_pitch + 1),
        add_end_token=add_end_token,
        slice_bars=slice_bars,
        pad_to_total_time=pad_to_total_time,
        steps_per_quarter=steps_per_quarter,
        quarters_per_bar=quarters_per_bar,
        max_tensors_per_notesequence=max_tensors_per_notesequence,
        presplit_on_time_changes=presplit_on_time_changes)

  def _to_tensors(self, note_sequence):
    def is_valid(note):
      if (self._valid_programs is not None and
          note.program not in self._valid_programs):
        return False
      return self._min_pitch <= note.pitch <= self._max_pitch
    notes = list(note_sequence.notes)
    del note_sequence.notes[:]
    note_sequence.notes.extend([n for n in notes if is_valid(n)])
    return super(OneHotMelodyConverter, self)._to_tensors(note_sequence)


class DrumsConverter(BaseNoteSequenceConverter):
  """Converter for legacy drums with either pianoroll or one-hot tensors.

  Inputs/outputs are either a "pianoroll"-like encoding of all possible drum
  hits at a given step, or a one-hot encoding of the pianoroll.

  The "roll" input encoding includes a final NOR bit (after the optional end
  token).

  Args:
    max_bars: Optional maximum number of bars per extracted drums, before
      slicing.
    slice_bars: Optional size of window to slide over raw Melodies after
      extraction.
    gap_bars: If this many bars or more follow a non-empty drum event, the
      drum track is ended. Disabled when set to 0 or None.
    pitch_classes: A collection of collections, with each sub-collection
      containing the set of pitches representing a single class to group by. By
      default, groups valid drum pitches into 9 different classes.
    add_end_token: Whether or not to add an end token. Recommended to be False
      for fixed-length outputs.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    roll_input: Whether to use a pianoroll-like representation as the input
      instead of a one-hot encoding.
    roll_output: Whether to use a pianoroll-like representation as the output
      instead of a one-hot encoding.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    presplit_on_time_changes: Whether to split NoteSequence on time changes
      before converting.
  """

  def __init__(self, max_bars=None, slice_bars=None, gap_bars=1.0,
               pitch_classes=None, add_end_token=False, steps_per_quarter=4,
               quarters_per_bar=4, pad_to_total_time=False, roll_input=False,
               roll_output=False, max_tensors_per_notesequence=5,
               presplit_on_time_changes=True):
    self._pitch_classes = pitch_classes or REDUCED_DRUM_PITCH_CLASSES
    self._pitch_class_map = {
        p: i for i, pitches in enumerate(self._pitch_classes) for p in pitches}

    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._slice_steps = self._steps_per_bar * slice_bars if slice_bars else None
    self._pad_to_total_time = pad_to_total_time
    self._roll_input = roll_input
    self._roll_output = roll_output

    self._drums_extractor_fn = functools.partial(
        mm.extract_drum_tracks,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=self._steps_per_bar * max_bars if max_bars else None,
        pad_end=True)

    num_classes = len(self._pitch_classes)

    self._pr_encoder_decoder = mm.PianorollEncoderDecoder(
        input_size=num_classes + add_end_token)
    # Use pitch classes as `drum_type_pitches` since we have already done the
    # mapping.
    self._oh_encoder_decoder = mm.MultiDrumOneHotEncoding(
        drum_type_pitches=[(i,) for i in range(num_classes)])

    output_depth = (num_classes if self._roll_output else
                    self._oh_encoder_decoder.num_classes) + add_end_token
    super(DrumsConverter, self).__init__(
        input_depth=(
            num_classes + 1 if self._roll_input else
            self._oh_encoder_decoder.num_classes) + add_end_token,
        input_dtype=np.bool,
        output_depth=output_depth,
        output_dtype=np.bool,
        end_token=output_depth - 1 if add_end_token else None,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors(self, note_sequence):
    """Converts NoteSequence to unique sequences."""
    try:
      quantized_sequence = mm.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (mm.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return [], []
    except (mm.BadTimeSignatureException, mm.NonIntegerStepsPerBarException,
            mm.NegativeTimeException) as e:
      return [], []

    new_notes = []
    for n in quantized_sequence.notes:
      if not n.is_drum:
        continue
      if n.pitch not in self._pitch_class_map:
        continue
      n.pitch = self._pitch_class_map[n.pitch]
      new_notes.append(n)
    del quantized_sequence.notes[:]
    quantized_sequence.notes.extend(new_notes)

    event_lists, unused_stats = self._drums_extractor_fn(quantized_sequence)

    if self._pad_to_total_time:
      for e in event_lists:
        e.set_length(len(e) + e.start_step, from_left=True)
        e.set_length(quantized_sequence.total_quantized_steps)
    if self._slice_steps:
      sliced_event_tuples = []
      for l in event_lists:
        for i in range(self._slice_steps, len(l) + 1, self._steps_per_bar):
          sliced_event_tuples.append(tuple(l[i - self._slice_steps: i]))
    else:
      sliced_event_tuples = [tuple(l) for l in event_lists]

    unique_event_tuples = list(set(sliced_event_tuples))
    unique_event_tuples = self._maybe_sample_outputs(unique_event_tuples)

    rolls = []
    oh_vecs = []
    for t in unique_event_tuples:
      if self._roll_input or self._roll_output:
        if self.end_token is not None:
          t_roll = list(t) + [(self._pr_encoder_decoder.input_size - 1,)]
        else:
          t_roll = t
        rolls.append(np.vstack([
            self._pr_encoder_decoder.events_to_input(t_roll, i).astype(np.bool)
            for i in range(len(t_roll))]))
      if not (self._roll_input and self._roll_output):
        labels = [self._oh_encoder_decoder.encode_event(e) for e in t]
        if self.end_token is not None:
          labels += [self._oh_encoder_decoder.num_classes]
        oh_vecs.append(np_onehot(
            labels,
            self._oh_encoder_decoder.num_classes + (self.end_token is not None),
            np.bool))

    if self._roll_input:
      input_seqs = [
          np.append(roll, np.expand_dims(np.all(roll == 0, axis=1), axis=1),
                    axis=1) for roll in rolls]
    else:
      input_seqs = oh_vecs

    output_seqs = rolls if self._roll_output else oh_vecs

    return input_seqs, output_seqs

  def _to_notesequences(self, samples):
    output_sequences = []
    for s in samples:
      if self._roll_output:
        if self.end_token is not None:
          end_i = np.where(s[:, self.end_token])
          if len(end_i):  # pylint: disable=g-explicit-length-test
            s = s[:end_i[0]]
        events_list = [frozenset(np.where(e)[0]) for e in s]
      else:
        s = np.argmax(s, axis=-1)
        if self.end_token is not None and self.end_token in s:
          s = s[:s.tolist().index(self.end_token)]
        events_list = [self._oh_encoder_decoder.decode_event(e) for e in s]
      # Map classes to exemplars.
      events_list = [
          frozenset(self._pitch_classes[c][0] for c in e) for e in events_list]
      track = mm.DrumTrack(
          events=events_list, steps_per_bar=self._steps_per_bar,
          steps_per_quarter=self._steps_per_quarter)
      output_sequences.append(track.to_sequence(velocity=80))
    return output_sequences


class TrioConverter(BaseNoteSequenceConverter):
  """Converts to/from 3-part (mel, drums, bass) multi-one-hot events.

  Extracts overlapping segments with melody, drums, and bass (determined by
  program number) and concatenates one-hot tensors from OneHotMelodyConverter
  and OneHotDrumsConverter. Takes the cross products from the sets of
  instruments of each type.

  Args:
    slice_bars: Optional size of window to slide over full converted tensor.
    gap_bars: The number of consecutive empty bars to allow for any given
      instrument. Note that this number is effectively doubled for internal
      gaps.
    max_bars: Optional maximum number of bars per extracted sequence, before
      slicing.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
  """

  class InstrumentType(object):
    UNK = 0
    MEL = 1
    BASS = 2
    DRUMS = 3
    INVALID = 4

  def __init__(
      self, slice_bars=None, gap_bars=2, max_bars=1024, steps_per_quarter=4,
      quarters_per_bar=4, max_tensors_per_notesequence=5):
    self._melody_converter = OneHotMelodyConverter(
        gap_bars=None, steps_per_quarter=steps_per_quarter,
        pad_to_total_time=True, presplit_on_time_changes=False,
        max_tensors_per_notesequence=None)
    self._drums_converter = DrumsConverter(
        gap_bars=None, steps_per_quarter=steps_per_quarter,
        pad_to_total_time=True, presplit_on_time_changes=False,
        max_tensors_per_notesequence=None)
    self._slice_bars = slice_bars
    self._gap_bars = gap_bars
    self._max_bars = max_bars
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar

    self._split_output_depths = (
        self._melody_converter.output_depth,
        self._melody_converter.output_depth,
        self._drums_converter.output_depth)
    output_depth = sum(self._split_output_depths)

    self._program_map = dict(
        [(i, TrioConverter.InstrumentType.MEL) for i in MEL_PROGRAMS] +
        [(i, TrioConverter.InstrumentType.BASS) for i in BASS_PROGRAMS])

    super(TrioConverter, self).__init__(
        input_depth=output_depth,
        input_dtype=np.bool,
        output_depth=output_depth,
        output_dtype=np.bool,
        end_token=False,
        presplit_on_time_changes=True,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors(self, note_sequence):
    try:
      quantized_sequence = mm.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (mm.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return [], []
    except (mm.BadTimeSignatureException, mm.NonIntegerStepsPerBarException,
            mm.NegativeTimeException):
      return [], []

    total_bars = int(
        np.ceil(quantized_sequence.total_quantized_steps / self._steps_per_bar))
    total_bars = min(total_bars, self._max_bars)

    # Assign an instrument class for each instrument, and compute its coverage.
    # If an instrument has multiple classes, it is considered INVALID.
    instrument_type = np.zeros(MAX_INSTRUMENT_NUMBER + 1, np.uint8)
    coverage = np.zeros((total_bars, MAX_INSTRUMENT_NUMBER + 1), np.bool)
    for note in quantized_sequence.notes:
      i = note.instrument
      if i > MAX_INSTRUMENT_NUMBER:
        tf.logging.warning('Skipping invalid instrument number: %d', i)
        continue
      inferred_type = (
          self.InstrumentType.DRUMS if note.is_drum else
          self._program_map.get(note.program, self.InstrumentType.INVALID))
      if not instrument_type[i]:
        instrument_type[i] = inferred_type
      elif instrument_type[i] != inferred_type:
        instrument_type[i] = self.InstrumentType.INVALID

      start_bar = note.quantized_start_step // self._steps_per_bar
      end_bar = int(np.ceil(note.quantized_end_step / self._steps_per_bar))

      if start_bar >= total_bars:
        continue
      coverage[start_bar:min(end_bar, total_bars), i] = True

    # Group instruments by type.
    instruments_by_type = collections.defaultdict(list)
    for i, type_ in enumerate(instrument_type):
      if type_ not in (self.InstrumentType.UNK, self.InstrumentType.INVALID):
        instruments_by_type[type_].append(i)
    if len(instruments_by_type) < 3:
      # This NoteSequence doesn't have all 3 types.
      return [], []

    # Encode individual instruments.
    # Set total time so that instruments will be padded correctly.
    note_sequence.total_time = (
        total_bars * self._steps_per_bar *
        60 / note_sequence.tempos[0].qpm / self._steps_per_quarter)
    encoded_instruments = {}
    for i in (instruments_by_type[self.InstrumentType.MEL] +
              instruments_by_type[self.InstrumentType.BASS]):
      _, t, _ = self._melody_converter.to_tensors(
          _extract_instrument(note_sequence, i))
      if t:
        encoded_instruments[i] = t[0]
      else:
        coverage[:, i] = False
    for i in instruments_by_type[self.InstrumentType.DRUMS]:
      _, t, _ = self._drums_converter.to_tensors(
          _extract_instrument(note_sequence, i))
      if t:
        encoded_instruments[i] = t[0]
      else:
        coverage[:, i] = False

    # Fill in coverage gaps up to self._gap_bars.
    og_coverage = coverage.copy()
    for j in range(total_bars):
      coverage[j] = np.any(
          og_coverage[
              max(0, j-self._gap_bars):min(total_bars, j+self._gap_bars) + 1],
          axis=0)

    # Take cross product of instruments from each class and compute combined
    # encodings where they overlap.
    seqs = []
    for grp in itertools.product(
        instruments_by_type[self.InstrumentType.MEL],
        instruments_by_type[self.InstrumentType.BASS],
        instruments_by_type[self.InstrumentType.DRUMS]):
      # Consider an instrument covered within gap_bars from the end if any of
      # the other instruments are. This allows more leniency when re-encoding
      # slices.
      grp_coverage = np.all(coverage[:, grp], axis=1)
      grp_coverage[:self._gap_bars] = np.any(coverage[:self._gap_bars, grp])
      grp_coverage[-self._gap_bars:] = np.any(coverage[-self._gap_bars:, grp])
      for j in range(total_bars - self._slice_bars + 1):
        if np.all(grp_coverage[j:j + self._slice_bars]):
          start_step = j * self._steps_per_bar
          end_step = (j + self._slice_bars) * self._steps_per_bar
          seqs.append(np.concatenate(
              [encoded_instruments[i][start_step:end_step] for i in grp],
              axis=-1))

    return seqs, seqs

  def _to_notesequences(self, samples):
    output_sequences = []
    dim_ranges = np.cumsum(self._split_output_depths)
    for s in samples:
      mel_ns = self._melody_converter.to_notesequences(
          [s[:, :dim_ranges[0]]])[0]
      bass_ns = self._melody_converter.to_notesequences(
          [s[:, dim_ranges[0]:dim_ranges[1]]])[0]
      drums_ns = self._drums_converter.to_notesequences(
          [s[:, dim_ranges[1]:]])[0]

      for n in bass_ns.notes:
        n.instrument = 1
        n.program = ELECTRIC_BASS_PROGRAM
      for n in drums_ns.notes:
        n.instrument = 9

      ns = mel_ns
      ns.notes.extend(bass_ns.notes)
      ns.notes.extend(drums_ns.notes)
      ns.total_time = max(
          mel_ns.total_time, bass_ns.total_time, drums_ns.total_time)
      output_sequences.append(ns)
    return output_sequences


def count_examples(examples_path, data_converter,
                   file_reader=tf.python_io.tf_record_iterator):
  """Counts the number of examples produced by the converter from files."""
  filenames = tf.gfile.Glob(examples_path)

  num_examples = 0

  for f in filenames:
    tf.logging.info('Counting examples in %s.', f)
    reader = file_reader(f)
    for item_str in reader:
      item = data_converter.str_to_item_fn(item_str)
      seqs, _, _ = data_converter.to_tensors(item)
      num_examples += len(seqs)
  tf.logging.info('Total examples: %d', num_examples)
  return num_examples


def get_dataset(config, tf_file_reader_class=tf.data.TFRecordDataset,
                num_threads=1, is_training=False):
  """Returns a Dataset object that encodes raw serialized NoteSequences."""
  examples_path = (
      config.train_examples_path if is_training else config.eval_examples_path)
  note_sequence_augmenter = (
      config.note_sequence_augmenter if is_training else None)
  data_converter = config.data_converter
  data_converter.is_training = is_training

  tf.logging.info('Reading examples from: %s', examples_path)

  files = tf.data.Dataset.list_files(examples_path)
  reader = files.apply(
      tf.contrib.data.parallel_interleave(
          tf_file_reader_class, cycle_length=num_threads))

  def _remove_pad_fn(padded_seq_1, padded_seq_2, length):
    if length.shape.ndims == 0:
      return padded_seq_1[0:length], padded_seq_2[0:length], length
    else:
      # Don't remove padding for hierarchical examples.
      return padded_seq_1, padded_seq_2, length

  dataset = reader
  if note_sequence_augmenter is not None:
    dataset = dataset.map(note_sequence_augmenter.tf_augment)
  dataset = (dataset
             .map(data_converter.tf_to_tensors,
                  num_parallel_calls=num_threads)
             .flat_map(
                 lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)))
             .map(_remove_pad_fn))

  return dataset


class TooLongError(Exception):
  """Exception for when an array is too long."""
  pass


def pad_with_element(nested_list, max_lengths, element):
  """Pads a nested list of elements up to `max_lengths`.

  For example, `pad_with_element([[0, 1, 2], [3, 4]], [3, 4], 5)` produces
  `[[0, 1, 2, 5], [3, 4, 5, 5], [5, 5, 5, 5]]`.

  Args:
    nested_list: A (potentially nested) list.
    max_lengths: The maximum length at each level of the nested list to pad to.
    element: The element to pad with at the lowest level. If an object, a copy
      is not made, and the same instance will be used multiple times.

  Returns:
    `nested_list`, padded up to `max_lengths` with `element`.

  Raises:
    TooLongError: If any of the nested lists are already longer than the
      maximum length at that level given by `max_lengths`.
  """
  if not max_lengths:
    return nested_list

  max_length = max_lengths[0]
  delta = max_length - len(nested_list)
  if delta < 0:
    raise TooLongError

  if len(max_lengths) == 1:
    return nested_list + [element] * delta
  else:
    return [pad_with_element(l, max_lengths[1:], element)
            for l in nested_list + [[] for _ in range(delta)]]


def pad_with_value(array, length, pad_value):
  """Pad numpy array so that its first dimension is length.

  Args:
    array: A 2D numpy array.
    length: Desired length of the first dimension.
    pad_value: Value to pad with.
  Returns:
    array, padded to shape `[length, array.shape[1]]`.
  Raises:
    TooLongError: If the array is already longer than length.
  """
  if array.shape[0] > length:
    raise TooLongError
  return np.pad(array, ((0, length - array.shape[0]), (0, 0)), 'constant',
                constant_values=pad_value)


class BaseHierarchicalConverter(BaseConverter):
  """Base class for data converters for hierarchical sequences.

  Output sequences will be padded hierarchically and flattened if `max_lengths`
  is defined. For example, if `max_lengths = [3, 2, 4]`, `end_token=5`, and the
  underlying `_to_tensors` implementation returns an example
  (before one-hot conversion) [[[1, 5]], [[2, 3, 5]]], `to_tensors` will
  convert it to:
    `[[1, 5, 0, 0], [5, 0, 0, 0],
      [2, 3, 5, 0], [5, 0, 0, 0],
      [5, 0, 0, 0], [5, 0, 0, 0]]`
  If any of the lengths are beyond `max_lengths`, the tensor will be filtered.

  Inheriting classes must implement the following abstract methods:
    -`_to_tensors`
    -`_to_items`
  """

  def __init__(self, input_depth, input_dtype, output_depth, output_dtype,
               end_token, max_lengths=None, max_tensors_per_item=None,
               str_to_item_fn=lambda s: s):
    self._max_lengths = [] if max_lengths is None else max_lengths
    super(BaseHierarchicalConverter, self).__init__(
        input_depth=input_depth,
        input_dtype=input_dtype,
        output_depth=output_depth,
        output_dtype=output_dtype,
        end_token=end_token,
        max_tensors_per_item=max_tensors_per_item,
        str_to_item_fn=str_to_item_fn,
        length_shape=(np.prod(max_lengths[:-1]),) if max_lengths else ())

  def to_tensors(self, item):
    """Converts to tensors and adds hierarchical padding, if needed."""
    unpadded_results = super(BaseHierarchicalConverter, self).to_tensors(item)
    if not self._max_lengths:
      return unpadded_results

    def _hierarchical_pad(input_, output):
      """Pad and flatten hierarchical inputs and outputs."""
      # Pad empty segments with end tokens and flatten hierarchy.
      input_ = nest.flatten(pad_with_element(
          input_, self._max_lengths[:-1],
          np_onehot([self.end_token], self.input_depth)))
      output = nest.flatten(pad_with_element(
          output, self._max_lengths[:-1],
          np_onehot([self.end_token], self.output_depth)))
      length = np.squeeze(np.array([len(x) for x in input_], np.int32))

      # Pad and concatenate flatten hierarchy.
      input_ = np.concatenate(
          [pad_with_value(x, self._max_lengths[-1], 0) for x in input_])
      output = np.concatenate(
          [pad_with_value(x, self._max_lengths[-1], 0) for x in output])

      return input_, output, length

    padded_results = []
    for i, o, _ in zip(*unpadded_results):
      try:
        padded_results.append(_hierarchical_pad(i, o))
      except TooLongError:
        continue

    return tuple(zip(*padded_results)) if padded_results else ([], [], [])


class SentenceConverter(BaseHierarchicalConverter):
  r"""Class for converting text data to (hierarchical) model input.

  Hierarchical data is handled using the `max_lengths` and `split_tokens` args.

  For example, if `split_tokens = ['\n', ' ']`, then the data will first be
  split into words (using the space as a delimiter) and then into lines (using
  the newline character as a delimiter). In this case, max_lengths must be
  length-3; the first entry would denote the maximum number of sentences,
  the second entry would denote the maximum number of words in a sentence and
  the third would be the maximum number of characters in a word.
  Any text with more than `max_lengths[0]` words in a sentence or
  `max_lengths[1]` characters in a word, etc. will be discarded. Note that the
  values in `max_length` should include end-of-sequence tokens in their counts.

  Args:
    max_lengths: (Optional) The maximum length at each level of the hierarchy,
      sized `[len(split_tokens) + 1]`.
    split_tokens: (Optional) Characters to use to split text into the hierarchy,
      sized `[len(max_lengths) - 1]`. These character tokenss will not be
      included in the converted output.
    valid_chars: Ordinal values of permissable characters.

  Raises:
    ValueError: If `max_length` or `split_tokens` is given but
      `len(max_length) != len(split_tokens) + 1`.
  """

  def __init__(self,
               max_lengths=None,
               split_tokens=None,
               valid_chars=range(32, 127)):
    self._char_map = {chr(c): i for i, c in enumerate(valid_chars)}
    self._chars = [chr(c) for c in valid_chars]
    depth = len(valid_chars) + 1
    self._split_tokens = [] if split_tokens is None else split_tokens

    super(SentenceConverter, self).__init__(
        input_depth=depth,
        input_dtype=np.bool,
        output_depth=depth,
        output_dtype=np.bool,
        end_token=len(valid_chars),
        max_lengths=max_lengths,
        max_tensors_per_item=None)

    if ((self._max_lengths or self._split_tokens) and
        len(self._max_lengths) != len(self._split_tokens) +1):
      raise ValueError(
          '`max_lengths` be 1 longer than `split_tokens`.'
          ' Got %d and %d.' % (len(self._max_lengths), len(self._split_tokens)))

  def _text_to_onehot(self, sentence):
    """Python method that converts `sentence` into tensors."""
    labels = []
    for c in sentence:
      if c not in self._char_map:
        tf.logging.warning('Skipped sentence with invalid char `%s`: %s',
                           c, sentence)
        return []
      labels.append(self._char_map[c])
    labels.append(self._end_token)
    seqs = np_onehot(labels, depth=self._output_depth)
    return seqs

  def _to_tensors(self, text):
    """Python method that converts `text` into tensors."""
    for token in self._split_tokens:
      text = nest.map_structure(lambda s: s.split(token), text)  # pylint: disable=cell-var-from-loop
    text = nest.map_structure(self._text_to_onehot, text)
    return [text], [text]

  def _to_items(self, samples):
    """Python method that decodes samples into list of sentences."""
    def _to_text(sample):
      tokens = np.argmax(sample, axis=-1).tolist()
      if self.end_token is not None and self.end_token in tokens:
        tokens = tokens[:tokens.index(self.end_token)]
      text = ''
      for t in tokens:
        assert t != self.end_token
        text += self._chars[t]
      return text

    sentences = []
    for sample in samples:
      if not self._split_tokens:
        sentences.append(_to_text(sample))
      elif len(self._split_tokens) == 1:
        sample = sample.reshape(self._max_lengths + [-1])
        words = [_to_text(segment) for segment in sample]
        sentences.append(self._split_tokens[0].join([w for w in words if w]))
      else:
        raise ValueError('Unsupported number of `split_tokens`.')

    return sentences
