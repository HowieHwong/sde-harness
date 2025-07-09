import numpy
from transformers import AutoModelForCausalLM, AutoTokenizer
import selfies as sf
from rdkit import Chem
import torch
import transformers
import crossover as co
import mutate as mu
# from main.molleo.mol_lm_utils import clean_edits
import re
from rdkit import Chem
import random
MINIMUM = 1e-10


class Llama3:
    def __init__(self):

        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda')

        self.task2description = {
                'qed': 'I have two molecules and their QED scores. The QED score measures the drug-likeness of the molecule.\n',
                'jnk3': 'I have two molecules and their JNK3 scores. The JNK3 score measures a molecular\'s biological activity against JNK3.\n\n',
                'drd2': 'I have two molecules and their DRD2 scores. The DRD2 score measures a molecule\'s biological activity against a biological target named the dopamine type 2 receptor (DRD2).\n\n',
                'gsk3b': 'I have two molecules and their GSK3$\beta$ scores. The GSK3$\beta$ score measures a molecular\'s biological activity against Glycogen Synthase Kinase 3 Beta.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'I have two molecules and their isomer scores. The isomer score measures a molecule\'s similarity in terms of atom counter to C9H10N2O2PF2Cl.\n\n',
                'perindopril_mpo': 'I have two molecules and their perindopril multiproperty objective scores. The perindopril multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to perindopril and number of aromatic rings.\n\n',
                'sitagliptin_mpo': 'I have two molecules and their sitagliptin multiproperty objective scores. The sitagliptin rediscovery score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to sitagliptin, TPSA score, LogP score and isomer score with C16H15F6N5O.\n\n',
                'ranolazine_mpo': 'I have two molecules and their ranolazine multiproperty objective scores. The ranolazine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to ranolazine, TPSA score LogP score and number of fluorine atoms.\n\n',
                'thiothixene_rediscovery': 'I have two molecules and their thiothixene rediscovery measures a molecule\'s Tanimoto similarity with thiothixene\'s SMILES to check whether it could be rediscovered.\n\n',
                'mestranol_similarity': 'I have two molecules and their mestranol similarity scores. The mestranol similarity score measures a molecule\'s Tanimoto similarity with Mestranol.\n\n',
                # 'fitness':"The following protein sequences are derived from directed evolution experiments aimed at improving protein's ability to package a DNA payload, i.e. for gene delivery. We are focusing on changes to a limited subset of amino acids within the sequence.\n "
                'fitness_AAV':"You will carry out a multi-round directed evolution experiment with the following protein sequence, aimed at improving protein's ability to package DNA payload, i.e, for gene delivery. We are focusing on changes to a limited subset of amino acids within the sequence.\n ",
                'fitness_GFP':"You will carry out a multi-round directed evolution experiment with the following protein sequence, aimed at improving protein's fluorescence properties as a biomarker. We are focusing on changes to a limited subset of amino acids within the sequence.\n "   
                }
        self.dataset_description = {
            'GB1': f'We are focusing on changes to a limited subset of amino acids within the sequence.\nThe provided subset protein sequences come from B1 DOMAIN OF STREPTOCOCCAL PROTEIN G, with sequence MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE. Each subset protein sequence represents specific amino acid substitutions at four key positions: 39, 40, 41, and 54, denoted using the single-letter amino acid code.\n',
            'TrpB': 'We are focusing on changes to a limited subset of amino acids within the sequence.\nThe provided subset protein sequences come from β-SUBUNIT OF TRYPTOPHAN SYNTHASE, with sequence MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMGKTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSVSAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIRLEHHHHHH. Each subset protein sequence represents specific amino acid substitutions at four key positions: 183, 184, 227, and 228, denoted using the single-letter amino acid code.',
            'Syn-3er7':f'The provided subset protein sequences come from CRYSTAL STRUCTURE OF NTF2-LIKE PROTEIN, with sequence TTLDRYFDLFDASRTDEKAFDDLISLFSDEITFVLNGQEQHGIDAWKQFVRMVFTANQDIKHMYAGWVPSETGDTMETRWAVCGKSADGSVFTQDGTDIARLNADGKIVYLANVPDDT',
            'AAV': 'The provided dataset comprises protein sequences derived from mutated variants of Adeno-Associated Virus (AAV) capsid proteins. These mutations are designed to study or improve the properties of AAV, such as its infectivity, tissue tropism, stability, or immune evasion. The dataset focuses on exploring how sequence variations influence AAV functionality, particularly in the context of gene delivery and therapeutic applications. The wild-type protein sequence in this dataset is DEEEIRTTNPVATEQYGSVSTNLQRGNR.\n',
            'GFP':'The provided dataset contains protein sequences derived from mutated variants of fluorescent proteins, specifically originating from the Green Fluorescent Protein (GFP) domain. These mutations are engineered or naturally occurring modifications of GFP to explore or enhance its fluorescent properties, stability, or other functional characteristics. The wild-type protein sequence in this dataset is SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
            
        }
        self.task2objective = {
                'qed': 'Please propose a new molecule that has a higher QED score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n',
                'jnk3': 'Please propose a new molecule that has a higher JNK3 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'drd2': 'Please propose a new molecule that has a higher DRD2 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'gsk3b': 'Please propose a new molecule that has a higher GSK3$\beta$ score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'Please propose a new molecule that has a higher isomer score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'perindopril_mpo': 'Please propose a new molecule that has a higher perindopril multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'sitagliptin_mpo': 'Please propose a new molecule that has a higher sitagliptin multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'ranolazine_mpo': 'Please propose a new molecule that has a higher ranolazine multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'thiothixene_rediscovery': 'Please propose a new molecule that has a higher thiothixene rediscovery score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'mestranol_similarity': 'Please propose a new molecule that has a higher mestranol similarity score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'fitness':"The following protein sequences are derived from directed evolution experiments aimed at improving protein's ability to package a DNA payload. We are focusing on changes to a limited subset of amino acids within the sequence.\n",
                'fitness_syn': '\nThe fitness score is a new score design for protein optimize. \nYou can either make crossover and mutations based on the given sequence or just propose a new sequence based on your knowledge.\n\n',
                }
        #         1. These sequences come from a directed evolution campaign targeting a specific enzymatic activity. Variants were generated using site-directed and combinatorial mutagenesis.  
        self.context = {
        'GFP':"""### Context:
        1. These sequences come from a directed evolution campaign targeting a specific enzymatic activity. Variants were generated using site-directed and combinatorial mutagenesis.  
        1.These capsid protein sequences were derived from a directed evolution campaign aimed at optimizing AAV's tissue-specific transduction and immune evasion. Variants were generated through site-directed and combinatorial mutagenesis.
        2. The limited number of sequences reflects experimental constraints, but they offer insight into promising regions of the fitness landscape.  
        3. Your proposal should focus on improving fitness while considering structural and functional plausibility.
        """,
        'AAV':"""### Context:
        1. These sequences come from a directed evolution campaign targeting enhanced AAV capsid functionality. Variants were generated using site-directed and combinatorial mutagenesis.  
        1.These capsid protein sequences were derived from a directed evolution campaign aimed at optimizing AAV's tissue-specific transduction and immune evasion. Variants were generated through site-directed and combinatorial mutagenesis.
        2. The limited number of sequences reflects experimental constraints, but they offer insight into promising regions of the fitness landscape.  
        3. Your proposal should focus on improving fitness while considering structural and functional plausibility.
        """
        }
        self.requirements = """\nYour output must only inclue: {\\box{$Protein}}.\n $Protein: Provide a new proposed valid protein sequence enclosed in \\box{} notation with same length as two protein sequences provided above. The sequence should have exactly same length as the given two sequences. 
        Ensure that the sequence is biologically plausible, reflects an understanding of protein fitness landscapes, and is optimized for the desired application.\nassistant
        """
        self.task=['fitness','fitness_syn']
        self.pipeline = transformers.pipeline(
            "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
    def edit(self, mating_tuples, mutation_rate, dataset='GB1'):

        task_definition = self.task2description[self.task[0]+'_'+dataset]
        # print(self.task[0])
        # print(self.task[0]+'_'+dataset)
        # exit()
        if dataset.__contains__('Syn'):
            task_objective = self.task2objective[self.task[1]] 
        else:
            task_objective = self.task2objective[self.task[0]]
        dataset_description = self.dataset_description[dataset]
        parent = []
        parent.append(random.choice(mating_tuples))
        parent.append(random.choice(mating_tuples))
        parent_mol = [t[1] for t in parent]
        parent_scores = [t[0] for t in parent]
        # length_parent_mol = len(parent_mol[0])
        context = self.context[dataset]

        try:
            mol_tuple = f'Here are the protein sequences has length of {len(parent_mol[0])} amino acids, and their fitness scores:'  
            for i in range(2):
                tu = f'\n{i+1}. Protein sequence: ' + ' '.join(parent_mol[i]) + ', Fitness score: ' + str(parent_scores[i])
                mol_tuple = mol_tuple + tu
            content = "system \nYou are a world-class assistant specializing in protein engineering, fitness optimization, and sequence design. Your expertise lies in analyzing sequence-function relationships, interpreting experimental data, and proposing rational modifications to optimize protein fitness. \nuser\n"
            prompt = content + task_definition + dataset_description + mol_tuple +'\n'+ task_objective + context + self.requirements
            # prompt = content + task_definition + dataset_description + mol_tuple +'\n'+ task_objective + self.context + self.requirements
            # print(prompt)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
            generation_config = self.model.generation_config
            # print(generation_config)
            # generation_config.max_tokens = 128
            if dataset=='GFP':
                # print("1")
                generation_config.max_new_tokens = 256
                generation_config.max_tokens = 256
            elif dataset == 'AAV':
                # print("2")
                generation_config.max_new_tokens = 64
                generation_config.max_tokens = 128
            # print(dataset)
            generation_config.num_beams = 1
            attention_mask = torch.tensor(input_ids!=128001)
            # print(input_ids.shape)
            # print('prompt: ', prompt)
            for retry in range(5):
                try:
                    if  dataset =='GFP' or dataset == 'AAV':
                        if dataset == 'GFP':
                            max_new_tokens = 256
                        elif dataset == 'AAV':
                            max_new_tokens = 64
                        # print(max_new_tokens)
                        outputs = self.pipeline(prompt, max_new_tokens = max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)[0]['generated_text'].removeprefix(prompt)
                        print(self.pipeline(prompt, max_new_tokens = max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)[0]['generated_text'])
                        exit()

                    else:
                        outputs = self.model.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config, pad_token_id=self.tokenizer.eos_token_id)
                        # print(outputs.shape[-1] - input_ids.shape[-1])
                        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].removeprefix(prompt)
                    # print('Retry ', retry , outputs)
                    # print(proposed_smiles)
                    # exit()
                    proposed_smiles = re.search(r'\\box\{(.*?)\}', outputs).group(1).replace('$', '').replace(' ', '')
                    # proposed_smiles = sanitize_smiles(proposed_smiles)
                    # print(proposed_smiles)
                    # exit()
                    assert proposed_smiles != None
                    assert len(proposed_smiles) == len(parent_mol[0])
                    new_child = proposed_smiles

                    break
                except Exception as e:
                    # print(f"{type(e).__name__} {e}")
                    continue
            return new_child
        except Exception as e:
            # print(f"{type(e).__name__} {e}")
            new_child = co.crossover_seq(parent_mol[0], parent_mol[1])
            if new_child is not None:
                new_child = mu.mutate_seq(new_child, mutation_rate)
            return new_child
def to(self, device):
    self.model.to(device) 
def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None
if __name__ == "__main__":
    model = Llama3()
    # model=model.to('cuda')
    print(model.edit([[0.001791394,"KWNA"],[0.007212719,"ARAF"],[2.31199298,"MRFG"],[0.022350843,'LDVA']],0.0))

