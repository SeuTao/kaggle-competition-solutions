# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name锛�     preparing
   Description :
   Author :       haxu
   date锛�          2019-06-18
-------------------------------------------------
   Change Activity:
                   2019-06-18:
-------------------------------------------------
"""
__author__ = 'haxu'

from molmod import *
import os
import numpy as np
from sklearn.model_selection import GroupKFold
from rdkit.Chem import AllChem
from xyz2mol import xyz2mol, read_xyz_file
from pathlib import Path
import pandas as pd
import pickle
import multiprocessing as mp
from sklearn import preprocessing
from rdkit import Chem
from utils import mol_from_axyz
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.EState as EState

from dscribe.descriptors import ACSF
from dscribe.core.system import System
import openbabel


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot


class Coupling:
    def __init__(self, id, contribution, index, type, value):
        self.id = id
        self.contribution = contribution
        self.index = index
        self.type = type
        self.value = value


class Graph:
    def __init__(self, molecule_name, smiles, axyz, node, edge, edge_index, coupling: Coupling):
        self.molecule_name = molecule_name
        self.smiles = smiles
        self.axyz = axyz
        self.node = node
        self.edge = edge
        self.edge_index = edge_index
        self.coupling = coupling

    def __str__(self):
        return f'graph of {self.molecule_name} -- smiles:{self.smiles}'


COUPLING_TYPE = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']
SYMBOL = ['H', 'C', 'N', 'O', 'F']
BOND_TYPE = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]


def gaussian_rbf(x, min_x, max_x, center_num):
    center_point = np.linspace(min_x, max_x, center_num)
    x_vec = np.exp(np.square(center_point - x))
    return x_vec


dist_min = 0.95860666
dist_max = 12.040386

ACSF_GENERATOR = ACSF(
    species=['H', 'C', 'N', 'O', 'F'],
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("xyz", "mol2")

atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71}  # Without fudge factor
fudge_factor = 0.05
atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}
electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}
electronegativity_square = {'H': 2.2 * 2.2, 'C': 2.55 * 2.55, 'N': 3.04 * 3.04, 'O': 3.44 * 3.44, 'F': 3.98 * 3.98}


def normal_dict(dict_input):
    min_value = min(dict_input.values())
    max_value = max(dict_input.values())
    for key in dict_input.keys():
        dict_input[key] = (dict_input[key] - min_value) / \
                          (max_value - min_value)
    return dict_input


atomic_mass = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984}
vanderwaalsradius = {'H': 120, 'C': 185, 'N': 154, 'O': 140, 'F': 135}
covalenzradius = {'H': 30, 'C': 77, 'N': 70, 'O': 66, 'F': 58}
ionization_energy = {'H': 13.5984, 'C': 11.2603, 'N': 14.5341, 'O': 13.6181, 'F': 17.4228}

atomic_mass = normal_dict(atomic_mass)
vanderwaalsradius = normal_dict(vanderwaalsradius)
covalenzradius = normal_dict(covalenzradius)
ionization_energy = normal_dict(ionization_energy)


def make_graph(name, gb_structure, gb_scalar_coupling):
    # ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type','scalar_coupling_constant']
    coupling_df = gb_scalar_coupling.get_group(name)

    # [molecule_name,atom_index,atom,x,y,z]
    df = gb_structure.get_group(name)
    df = df.sort_values(['atom_index'], ascending=True)
    a = df.atom.values.tolist()
    xyz = df[['x', 'y', 'z']].values

    mol = mol_from_axyz(a, xyz)
    mol_op = openbabel.OBMol()
    obConversion.ReadFile(mol_op, f'../input/champs-scalar-coupling/structures/{name}.xyz')

    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    feature = factory.GetFeaturesForMol(mol)

    num_atom = mol.GetNumAtoms()
    symbol = np.zeros((num_atom, len(SYMBOL)), np.uint8)  # category
    acceptor = np.zeros((num_atom, 1), np.uint8)
    donor = np.zeros((num_atom, 1), np.uint8)
    aromatic = np.zeros((num_atom, 1), np.uint8)
    hybridization = np.zeros((num_atom, len(HYBRIDIZATION)), np.uint8)
    num_h = np.zeros((num_atom, 1), np.float32)  # real
    atomic = np.zeros((num_atom, 1), np.float32)

    # new features
    degree = np.zeros((num_atom, 1), np.uint8)
    formalCharge = np.zeros((num_atom, 1), np.float32)
    chiral_tag = np.zeros((num_atom, 1), np.uint8)
    crippen_contribs = np.zeros((num_atom, 2), np.float32)
    tpsa = np.zeros((num_atom, 1), np.float32)
    labute_asac = np.zeros((num_atom, 1), np.float32)
    gasteiger_charges = np.zeros((num_atom, 1), np.float32)
    esataindices = np.zeros((num_atom, 1), np.float32)
    atomic_radiuss = np.zeros((num_atom, 1), np.float32)
    electronegate = np.zeros((num_atom, 1), np.float32)
    electronegate_sqre = np.zeros((num_atom, 1), np.float32)
    mass = np.zeros((num_atom, 1), np.float32)
    van = np.zeros((num_atom, 1), np.float32)
    cov = np.zeros((num_atom, 1), np.float32)
    ion = np.zeros((num_atom, 1), np.float32)

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        atom_op = mol_op.GetAtomById(i)
        symbol[i] = one_hot_encoding(atom.GetSymbol(), SYMBOL)
        aromatic[i] = atom.GetIsAromatic()
        hybridization[i] = one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION)
        num_h[i] = atom.GetTotalNumHs(includeNeighbors=True)
        atomic[i] = atom.GetAtomicNum()

        degree[i] = atom.GetTotalDegree()
        formalCharge[i] = atom.GetFormalCharge()
        chiral_tag[i] = int(atom.GetChiralTag())

        crippen_contribs[i] = rdMolDescriptors._CalcCrippenContribs(mol)[i]
        tpsa[i] = rdMolDescriptors._CalcTPSAContribs(mol)[i]
        labute_asac[i] = rdMolDescriptors._CalcLabuteASAContribs(mol)[0][i]
        gasteiger_charges[i] = atom_op.GetPartialCharge()
        esataindices[i] = EState.EStateIndices(mol)[i]
        atomic_radiuss[i] = atomic_radius[atom.GetSymbol()]
        electronegate[i] = electronegativity[atom.GetSymbol()]
        electronegate_sqre[i] = electronegativity_square[atom.GetSymbol()]
        mass[i] = atomic_mass[atom.GetSymbol()]
        van[i] = vanderwaalsradius[atom.GetSymbol()]
        cov[i] = covalenzradius[atom.GetSymbol()]
        ion[i] = ionization_energy[atom.GetSymbol()]

    for t in range(0, len(feature)):
        if feature[t].GetFamily() == 'Donor':
            for i in feature[t].GetAtomIds():
                donor[i] = 1
        elif feature[t].GetFamily() == 'Acceptor':
            for i in feature[t].GetAtomIds():
                acceptor[i] = 1

    num_edge = num_atom * num_atom - num_atom
    edge_index = np.zeros((num_edge, 2), np.uint32)
    bond_type = np.zeros((num_edge, len(BOND_TYPE)), np.uint32)
    distance = np.zeros((num_edge, 1), np.float32)
    angle = np.zeros((num_edge, 1), np.float32)

    norm_xyz = preprocessing.normalize(xyz, norm='l2')

    ij = 0
    for i in range(num_atom):
        for j in range(num_atom):
            if i == j: continue
            edge_index[ij] = [i, j]

            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond_type[ij] = one_hot_encoding(bond.GetBondType(), BOND_TYPE)

            distance[ij] = np.linalg.norm(xyz[i] - xyz[j])
            angle[ij] = (norm_xyz[i] * norm_xyz[j]).sum()

            ij += 1

    xyz = xyz * 1.889726133921252

    atom = System(symbols=a, positions=xyz)
    acsf = ACSF_GENERATOR.create(atom)

    l = []
    for item in coupling_df[['atom_index_0', 'atom_index_1']].values.tolist():
        i = edge_index.tolist().index(item)
        l.append(i)

    l = np.array(l)

    coupling_edge_index = np.concatenate([coupling_df[['atom_index_0', 'atom_index_1']].values, l.reshape(len(l), 1)],
                                         axis=1)

    coupling = Coupling(coupling_df['id'].values,
                        coupling_df[['fc', 'sd', 'pso', 'dso']].values,
                        coupling_edge_index,
                        np.array([COUPLING_TYPE.index(t) for t in coupling_df.type.values], np.int32),
                        coupling_df['scalar_coupling_constant'].values,
                        )

    graph = Graph(
        name,
        Chem.MolToSmiles(mol),
        [a, xyz],
        [acsf, symbol, acceptor, donor, aromatic, hybridization, num_h, atomic, degree, formalCharge, chiral_tag,
         crippen_contribs, tpsa, labute_asac, gasteiger_charges, esataindices, atomic_radiuss, electronegate,
         electronegate_sqre, mass, van, cov, ion],
        [bond_type, distance, angle, ],
        edge_index,
        coupling,
    )

    return graph


if __name__ == '__main__':
    df_structure = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
    df_train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
    df_test = pd.read_csv('../input/champs-scalar-coupling/test.csv')
    df_test['scalar_coupling_constant'] = 0
    df_scalar_coupling = pd.concat([df_train, df_test])
    df_scalar_coupling_contribution = pd.read_csv('../input/champs-scalar-coupling/scalar_coupling_contributions.csv')
    df_scalar_coupling = pd.merge(df_scalar_coupling, df_scalar_coupling_contribution,
                                  how='left',
                                  on=['molecule_name', 'atom_index_0', 'atom_index_1', 'atom_index_0', 'type'])

    gb_scalar_coupling = df_scalar_coupling.groupby('molecule_name')
    gb_structure = df_structure.groupby('molecule_name')

    molecule_names = df_scalar_coupling.molecule_name.unique()

    g = make_graph('dsgdb9nsd_000003', gb_structure, gb_scalar_coupling)

    print(g.node)
    print(g.edge)
    print(g.smiles)

    param = []


    def do_one(p):
        molecule_name, graph_file = p
        g = make_graph(molecule_name, gb_structure, gb_scalar_coupling)
        with open(graph_file, 'wb') as f:
            pickle.dump(g, f)


    for i, molecule_name in enumerate(molecule_names):
        graph_file = f'../input/graph/{molecule_name}.pickle'
        p = (molecule_name, graph_file)
        param.append(p)

    print('load done.')

    pool = mp.Pool(processes=55)
    _ = pool.map(do_one, param)

    pool.close()
    pool.join()
