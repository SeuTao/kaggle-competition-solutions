# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       haxu
   date：          2019-06-20
-------------------------------------------------
   Change Activity:
                   2019-06-20:
-------------------------------------------------
"""
__author__ = 'haxu'

import os
import sys
import numpy as np
import copy
import pandas as pd
from xyz2mol import get_atomicNumList, xyz2mol


def mol_from_axyz(symbol, xyz):
    charged_fragments = True
    quick = True
    charge = 0
    atom_no = get_atomicNumList(symbol)
    mol = xyz2mol(atom_no, xyz, charge, charged_fragments, quick)
    return mol


def read_list_from_file(list_file, comment='#'):
    with open(list_file) as f:
        lines = f.readlines()

    strings = []
    for line in lines:
        if comment is not None:
            s = line.split(comment, 1)[0].strip()
        else:
            s = line.strip()

        if s != '':
            strings.append(s)

    return strings


def read_champs_xyz(xyz_file):
    line = read_list_from_file(xyz_file, comment=None)
    num_atom = int(line[0])
    xyz = []
    symbol = []
    for n in range(num_atom):
        l = line[1 + n]
        l = l.replace('\t', ' ').replace('  ', ' ')
        l = l.split(' ')
        symbol.append(l[0])
        xyz.append([float(l[1]), float(l[2]), float(l[3]), ])

    return symbol, xyz


def map_atom_info(df):
    df_structures = pd.read_csv('../input/champs-scalar-coupling/structures_621.csv')
    for atom_idx in range(2):
        df = pd.merge(df, df_structures, how='left',
                      left_on=['molecule_name', f'atom_index_{atom_idx}'],
                      right_on=['molecule_name', 'atom_index'])

        df = df.drop('atom_index', axis=1)
        df = df.rename(columns={'atom': f'atom_{atom_idx}',
                                'x': f'x_{atom_idx}',
                                'y': f'y_{atom_idx}',
                                'z': f'z_{atom_idx}'})

    atom_count = df_structures.groupby(['molecule_name', 'atom']).size().unstack(fill_value=0)
    df = pd.merge(df, atom_count, how='left', left_on='molecule_name', right_on='molecule_name')
    del df_structures
    return df


def search_neighbors(mol, atom_list):
    '''
    Find the path from the first atom to the secend atom,
    and then add the 1-step-neighbors of this path.
    '''
    the_path_between_two_atom = [atom_list[0]]
    depth = 0
    # find the path between two atom
    the_path_between_two_atom = each_atom_in_searching(
        mol, atom_list[0], atom_list[1], depth, the_path_between_two_atom)
    # add the neighbors of the atoms in the path
    the_neighbors_of_the_path = copy.deepcopy(
        the_path_between_two_atom)
    for i in the_path_between_two_atom:
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            # if edge is not none and j is not in the set, j is a new node
            if (e_ij is not None) and (j not in the_neighbors_of_the_path):
                the_neighbors_of_the_path.append(j)
    return the_neighbors_of_the_path, the_path_between_two_atom


def each_atom_in_searching(mol, atom, end_atom, depth, path):
    if depth >= 3:
        return path
    neighbors_list_of_this_atom = []
    for j in range(mol.GetNumAtoms()):
        e_ij = mol.GetBondBetweenAtoms(atom, j)
        # if edge is not none and j is not in the set, j is a new node
        if (e_ij is not None) and (j not in path):
            neighbors_list_of_this_atom.append(j)

    if end_atom not in neighbors_list_of_this_atom:
        depth += 1
        for j in neighbors_list_of_this_atom:
            # keep path
            path_j = copy.deepcopy(path)
            path_j.append(j)
            path_j = each_atom_in_searching(
                mol, j, end_atom, depth, path_j)
            if path_j is not None and end_atom in path_j:
                return path_j
    else:
        path.append(end_atom)
        return path
