import argparse
import math
import numpy as np

import os

import torch
from torch import optim, nn
import torch.nn.functional as F
from tqdm import tqdm
from .mol_lm_utils import get_SMILES_list, get_description_list, load_language_molecule_and_edit_models, clip_loss_for_edit, evaluate_SMILES_list
from .mol_lm_utils import prepare_text_tokens, clean_edits
from MoleculeSTM.utils import prepare_text_tokens, get_molecule_repr_MoleculeSTM, freeze_network
from rdkit import Chem
from utils import get_fp_scores


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def mean_pooling(token_embeddings, attention_mask):
    attention_mask = ~attention_mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask


class MolCLIP(nn.Module):
    def __init__(self):
        super(MolCLIP, self).__init__()

        self.args = self.parse_args()
        self.args.use_noise_for_init = False


        device = torch.device("cuda:" + str(self.args.device)) \
            if torch.cuda.is_available() else torch.device("cpu")

        self.load_modules()

        np.random.seed(self.args.seed)

        torch.random.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        device = torch.device("cuda:" + str(self.args.device)) \
            if torch.cuda.is_available() else torch.device("cpu")

        self.device=device
        self.task2description = { 'jnk3': 'This molecule inhibits JNK3.',
                                  "gsk3b": 'This molecule inhibits GSK3B.',
                                  "drd2": 'This molecule inhibits DRD2.',
                                'perindopril_mpo': "This molecule looks like Perindopril.",
                                'sitagliptin_mpo': "This molecule looks like Sitagliptin.",
                                'mestranol_similarity': "This molecule looks like Mestranol.",
                                'thiothixene_rediscovery': "This molecule looks like Thiothixene.",
                                 'Isomers_C9H10N2O2PF2Cl': "This molecule has the atoms C9H10N2O2PF2Cl.",
                                 }
        self.l2_lambda_list = [1]
        #self.l2_lambda_list = [0, 0.01, 0.1, 1, 10]
        self.task = None

    def __name__(self):
        return "mol clip"

    def load_modules(self, load_molmodel=False):
        text_model, text_tokenizer, text_dim, molecule_model, MegaMolBART_wrapper, molecule_dim, \
            text2latent, mol2latent, generation2MoleculeSTM, MoleculeSTM2generation = load_language_molecule_and_edit_models(self.args, load_molmodel=load_molmodel)

        device = self.args.device
        self.text_model = text_model.to(device)
        self.molecule_model = molecule_model.to(device)
        self.text2latent = text2latent.to(device)
        self.mol2latent = mol2latent.to(device)
        self.generation2MoleculeSTM = generation2MoleculeSTM.to(device)
        self.MoleculeSTM2generation = MoleculeSTM2generation.to(device)
        self.text_model.eval()
        self.molecule_model.eval()
        self.text2latent.eval()
        self.mol2latent.eval()
        self.generation2MoleculeSTM.eval()
        self.MoleculeSTM2generation.eval()

        self.MegaMolBART_wrapper = MegaMolBART_wrapper
        self.text_tokenizer = text_tokenizer
        
        self.temperature = nn.parameter.Parameter(torch.tensor(0.0), requires_grad=True)



    def get_text_repr(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            device=self.device, description=text, tokenizer=self.text_tokenizer, max_seq_len=512)
        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        text_repr = text_output["pooler_output"]
        text_repr = self.text2latent(text_repr)
        return text_repr

    def forward(self, molecule_data, text_prompt):
        molecule_data = list(molecule_data) # for SMILES_list
        molecule_repr = get_molecule_repr_MoleculeSTM(
                molecule_data, mol2latent=self.mol2latent,
                molecule_type="SMILES", MegaMolBART_wrapper=self.MegaMolBART_wrapper)

        text_repr = self.get_text_repr([text_prompt])
        output = self.do_CL_eval(text_repr, molecule_repr)
        return output 

    def do_CL_eval(self, X, Y):
        """
        X is shape 1 x d
        Y is shape B x d
        want output B x 1 --> sum(X * Y)
        """
        X = F.normalize(X, dim=-1)

        Y = F.normalize(Y, dim=-1)
        Y = Y.repeat(X.shape[0], 1)

        logits = torch.sum(X * Y, dim=1)
    
        return logits

    def check_edit(self, SMILES, text):
        text_list = [text]
        text_tokens_ids, text_masks = prepare_text_tokens(
            device=self.device, description=text_list, tokenizer=self.text_tokenizer, max_seq_len=self.args.max_seq_len)

        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        text_repr = text_output["pooler_output"]
        text_repr = self.text2latent(text_repr)
    
        first_and_second_SMILES_list = []
    
        latent_code_init, pad_mask_init = self.MegaMolBART_wrapper.smileslist2embedding([SMILES])  # [pad, B, d], [pad, B]
        first_and_second_SMILES_list.append(SMILES)
    
        regenerated_mols = self.MegaMolBART_wrapper.inverse_transform([latent_code_init], pad_mask_init.bool().cuda(), k=1, sanitize=True)
        first_and_second_SMILES_list.append(regenerated_mols[0])
    

        result_SMILES_list_one_pair, result_eval_list_one_pair = [], []
        
        if self.args.use_noise_for_init:
            print("Use random noise for init")
            device='cuda'
            random_noise = torch.randn(latent_code_init.size()).to(device)
        
        for l2_lambda in self.l2_lambda_list:
            print("lambda", l2_lambda)
            current_SMILES_list = [first_and_second_SMILES_list[0]] + [first_and_second_SMILES_list[1]]
            if self.args.use_noise_for_init:
                latent = latent_code_init.detach().clone() + 0.1 * random_noise
            else:
                latent = latent_code_init.detach().clone()
            pad_mask = pad_mask_init.detach().clone()
            latent.requires_grad = True
            optimizer = optim.Adam([latent], lr=self.args.lr)
            
            L = range(self.args.epochs)
    
            for i in L:
                t = i / self.args.epochs
                lr = get_lr(t, self.args.lr)
                optimizer.param_groups[0]["lr"] = lr
    
                molecule_repr_generation = mean_pooling(latent, pad_mask) # [B, d]
                if self.args.normalize:
                    molecule_repr_generation = F.normalize(molecule_repr_generation, dim=-1)
                molecule_repr_MoleculeSTM = self.generation2MoleculeSTM(molecule_repr_generation)
    
                clip_loss_ = clip_loss_for_edit(molecule_repr_MoleculeSTM, text_repr)

                l2_loss_ =  l2_lambda * ((latent_code_init - latent) ** 2).mean()
    
                loss = clip_loss_ + l2_loss_
    
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
    
            generated_mols = self.MegaMolBART_wrapper.inverse_transform([latent], pad_mask.bool().cuda(), k=1, sanitize=True)
            current_SMILES_list.append(generated_mols[0])
            try:
                fp_score = get_fp_scores([generated_mols[0]], SMILES)
                print(f"fp score:: {fp_score}")
            except:
                pass
            result_SMILES_list_one_pair.append([text] + current_SMILES_list + ['{}'.format(l2_lambda)])
    
            current_result_list, oracle_vals = evaluate_SMILES_list(current_SMILES_list, text)
            result_eval_list_one_pair.append(current_result_list)
        
        result_eval_list_one_pair = np.array(result_eval_list_one_pair)
        result_eval_list_one_pair = np.any(result_eval_list_one_pair, axis=0, keepdims=True)
        return result_SMILES_list_one_pair, result_eval_list_one_pair, oracle_vals


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--verbose", type=int, default=1)

        ########## for editing ##########
        parser.add_argument("--use_noise_for_init", dest="use_noise_for_init", action="store_true")
        parser.add_argument("--no_noise_for_init", dest="use_noise_for_init", action="store_false")
        parser.set_defaults(use_noise_for_init=False)
        parser.add_argument('--normalize', dest='normalize', action='store_true')
        parser.add_argument('--no_normalize', dest='normalize', action='store_false')
        parser.set_defaults(normalize=True)

        parser.add_argument("--dataspace_path", type=str, default="../data")
        parser.add_argument("--SSL_emb_dim", type=int, default=256)
        parser.add_argument("--max_seq_len", type=int, default=512)

        ########## for MoleculeSTM ##########
        parser.add_argument("--MoleculeSTM_model_dir", type=str, default="/scratch/ssd004/scratch/mskrt/huggingface_models/MoleculeSTM/demo/demo_checkpoints_SMILES")
        parser.add_argument("--MoleculeSTM_molecule_type", type=str, default="SMILES", choices=["SMILES", "Graph"])

        ########## for MegaMolBART ##########
        parser.add_argument("--MegaMolBART_generation_model_dir", type=str, default="/scratch/ssd004/scratch/mskrt/huggingface_models/MoleculeSTM/megamolbart/models/megamolbart/checkpoints")
        parser.add_argument("--vocab_path", type=str, default="/h/mskrt/language_guided_genetic_algorithms/MoleculeSTM/bart_vocab.txt")

        ########## for MoleculeSTM and generation projection ##########
        parser.add_argument("--language_edit_model_dir", type=str, default="/scratch/ssd004/scratch/mskrt/huggingface_models/MoleculeSTM/demo/demo_checkpoints_SMILES")   

        ########## for editing ##########
        parser.add_argument("--lr_rampup", type=float, default=0.05)
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--epochs", type=int, default=30)

        args, _ = parser.parse_known_args()
        return args

    def edit(self, smiles_list, description=None, l2=None, epoch=None): 
        if l2 != None:
            self.l2_lambda_list = [l2]
        if epoch != None:
            self.args.epochs = epoch
        task = self.task
        if description == None:
            description = self.task2description[task[0]]
        print("\n\n\nstart editing\n\n\n")
        print("===== for description {} =====".format(description))
        result_SMILES_list, result_acc_list = [], []
        editted_molecules = []
        before, after = [], []
        for i, MOL in enumerate(smiles_list):
            SMILES = Chem.MolToSmiles(MOL)
            print("===== for SMILES {} =====".format(SMILES))
            result_SMILES_list_, _, oracle_vals = self.check_edit(SMILES, description)
            if len(oracle_vals) > 0: 
                for val_ in oracle_vals:
                    before.append(val_[0])
                    after.append(val_[1])
            generated_smiles = [output[3] for output in result_SMILES_list_]
            editted_molecules.extend(generated_smiles)
        editted_molecules = clean_edits(editted_molecules)
        print("before", before)
        print("after", after)
        num_improved = 0
        average_improvement = 0
        for i in range(len(before)):
            if after[i] > before[i]:
                num_improved += 1
                average_improvement += after[i] - before[i]
        
        print("num improved", num_improved/max(1, len(after)))
        num_improved = max(1, num_improved)
        print("average improvement", average_improvement/num_improved)
        return editted_molecules

if __name__ == "__main__":
    smi_list = ["C=CC1=CC=C([N+](=O)[O-])C=C1",
            "C[C@@H]1C(C(CO)N(N=N)C2=CC=CC([N+](=O)[O-])=C2)C1[C@H]([NH3+])N1N=NC2C1=NC1=C2C=C([N+](=O)[O-])C=C1",
            "O=[N+]([O-])C1=CC2=C(C=C1)N=NN2C=CO", 
            "C1=CN(C2=CC=C(C3=CC=NC(NC4=CCCC4)=C3)N=C2)C=CC1",
            "O=[N+]([O-])C=CC1=NC=NC2=CC(NC3=CCCC=C3)=CC=C21",
            "CNC=CNC1=NC=NC2=C1C=CC(NC1=CCCC=C1)=C2",
            "CC1=C(C)C=C(NC2=CC(F)=CC=C2)N=C1",
            "C=C(Cl)C1=CC(NC(=O)[N+](=O)[O-])=CC([N+](=O)[O-])=C1",
            "C[NH+]1CCCC[C@@H]1C1=NC(CN)=NN1C=CC=CN1N=NC2=CC=C([N+](=O)[O-])C=C21",
            "O=C([O-])NC1=CC(C=C[N+](=O)[O-])=CC1",
            "CN1N=NC2=C1C=C([N+](=O)[C@H](O)C1=C(F)C(C3=CCOC(C(=S)N4N=NC5=CC=C([N+](=O)[O-])C=C54)=C3)C=C1)C=C2",
            "O=[N+]([O-])C1=CC=C(OCC(=S)ON2N=NC3=CC=C([N+](=O)[O-])C=C32)C=C1",
            "O=[N+]([O-])C1=CC=C2N=NN(C=CC3=CC([N+](=O)[O-])C=CC3[N+](=O)[O-])C2=C1",
            ]
    #smi_list = []
    #with open("/h/mskrt/language_guided_genetic_algorithms/JANUS/tests/molopt_startpop_seed0.txt") as f:
    #    for line in f:
    #        smi_list.append(line[:-1])

    smi_list = [Chem.MolFromSmiles(m) for m in smi_list]
    edit_model = MolCLIP()
    edit_model.task = ["Isomers_C9H10N2O2PF2Cl"]
    edit_model.edit(smi_list, description=None)
