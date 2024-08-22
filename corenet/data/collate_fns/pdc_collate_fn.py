import argparse
import subprocess
import tempfile
import os
from typing import Dict, List, Mapping, Optional, Union
import random

import torch
from torch import Tensor
from torch.nn import functional

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.vectors import rotmat, Vector
from Bio.PDB.transform import Transform

from corenet.data.collate_fns import COLLATE_FN_REGISTRY, collate_functions

@COLLATE_FN_REGISTRY.register(name="byteformer_pdc_collate_fn")
def byteformer_pdc_collate_fn(
    batch: List[Mapping[str, Tensor]], opts: argparse.Namespace
) -> Mapping[str, Tensor]:
    """
    Apply augmentations specific to ByteFormer PDC file training, then perform padded collation.

    Args:
        batch: The batch of PDC file data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    batch = apply_pdc_augmentation(batch, opts)
    batch = apply_padding(batch, opts)
    batch = collate_functions.pytorch_default_collate_fn(batch, opts)
    return batch

def apply_pdc_augmentation(
    batch: List[Mapping[str, Tensor]],
    opts: argparse.Namespace,
) -> List[Mapping[str, Tensor]]:
    """
    Apply PDC-specific augmentations to each batch element.

    Args:
        batch: The batch of PDC file data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    if getattr(opts, "pdc_augmentation.enable", False):
        for i, elem in enumerate(batch):
            with tempfile.TemporaryDirectory() as tmpdir:
                input_pdc = os.path.join(tmpdir, "input.pdc")
                input_pdb = os.path.join(tmpdir, "input.pdb")
                output_pdb = os.path.join(tmpdir, "output.pdb")
                output_pdc = os.path.join(tmpdir, "output.pdc")

                # Save the input PDC file
                with open(input_pdc, 'wb') as f:
                    f.write(elem['samples'].tobytes())

                # Convert PDC to PDB
                subprocess.run(['pdd', input_pdc, input_pdb], check=True)

                # Load PDB, apply augmentations, and save
                structure = PDBParser().get_structure("protein", input_pdb)
                structure = apply_random_rotation(structure, opts)
                structure = apply_random_translation(structure, opts)

                io = PDBIO()
                io.set_structure(structure)
                io.save(output_pdb)

                # Convert back to PDC
                subprocess.run(['pdc', output_pdb, output_pdc], check=True)

                # Read the augmented PDC file
                with open(output_pdc, 'rb') as f:
                    augmented_bytes = f.read()

                batch[i]['samples'] = torch.frombuffer(augmented_bytes, dtype=torch.uint8)

    return batch

def apply_random_rotation(structure, opts):
    """
    Apply a random rotation to the protein structure.

    Args:
        structure: A Bio.PDB Structure object.
        opts: The global arguments.

    Returns:
        The rotated structure.
    """
    angle = random.uniform(0, 360)
    axis = Vector(random.gauss(0,1), random.gauss(0,1), random.gauss(0,1)).normalized()
    rotation = rotmat(axis, angle)
    
    for atom in structure.get_atoms():
        atom.transform(rotation, Vector(0,0,0))
    
    return structure

def apply_random_translation(structure, opts):
    """
    Apply a random translation to the protein structure.

    Args:
        structure: A Bio.PDB Structure object.
        opts: The global arguments.

    Returns:
        The translated structure.
    """
    translation = Vector(random.gauss(0,5), random.gauss(0,5), random.gauss(0,5))
    
    for atom in structure.get_atoms():
        atom.transform(rotmat(Vector(1,0,0), 0), translation)
    
    return structure

def apply_padding(
    batch: List[Mapping[str, Union[Dict[str, Tensor], Tensor]]],
    opts: argparse.Namespace,
    key: Optional[str] = None,
) -> List[Mapping[str, Tensor]]:
    """
    Apply padding to make samples the same length.

    Args:
        batch: The batch of data.
        opts: The global arguments.
        key: The key of the sample element to pad. If @key is None, the entry
            is assumed to be a tensor.

    Returns:
        The modified batch of size [batch_size, padded_sequence_length, ...].
    """
    def get_entry(
        entry: Union[Dict[str, Tensor], Tensor], key: Optional[str]
    ) -> Tensor:
        if isinstance(entry, dict):
            return entry[key]
        if key is not None:
            raise ValueError(f"Key should not be specified if entries are not dicts.")
        return entry

    if get_entry(batch[0]["samples"], key).dim() != 1:
        # Padding only applies to 1d tensors.
        return batch
    padding_idx = getattr(opts, "model.classification.byteformer.padding_index")
    padded_seq_len = max(get_entry(be["samples"], key).shape[0] for be in batch)
    for elem in batch:
        sample = get_entry(elem["samples"], key)
        sample = functional.pad(
            sample, (0, padded_seq_len - sample.shape[0]), value=padding_idx
        )
        if isinstance(elem["samples"], dict):
            elem["samples"][key] = sample
        else:
            elem["samples"] = sample
    return batch
