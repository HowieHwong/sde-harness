import openai
import re
from rdkit import Chem
import main.molleo_multi.crossover as co, main.molleo_multi.mutate as mu
openai.api_type = "azure"
openai.api_base = 
openai.api_version = "2023-07-01-preview"
openai.api_key = 
import random
MINIMUM = 1e-10

def query_LLM(question, model="gpt-4", temperature=0.0):
    message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]

    prompt1 = question
    message.append({"role": "user", "content": prompt1})

    params = {
        "engine": , #OPENAI_engine, Please use your own
        "max_tokens": 2048,
        "temperature": temperature,
        "messages": message
    }

    for retry in range(3):
        try:
            response = openai.ChatCompletion.create(**params)["choices"][0]["message"]["content"]
            message.append({"role": "assistant", "content": response})
            break
        except Exception as e:
            print(f"{type(e).__name__} {e}")


    print("=>")
    return message, response

class GPT4:
    def __init__(self):
        self.task2description_mul = {
                '1': 'I have two molecules and their QED, SA (Synthetic Accessibility), JNK3 (biological activity against the kinase JNK3) scores.\n\n',
                '2': 'I have two molecules and their QED, SA (Synthetic Accessibility), GSK3$\beta$ (biological activity against Glycogen Synthase Kinase 3 Beta) scores.\n\n',
                '3': 'I have two molecules and their QED, SA (Synthetic Accessibility), JNK3 (biological activity against the kinase JNK3), GSK3$\beta$ (biological activity against Glycogen Synthase Kinase 3 Beta), DRD2 (biological activity against a biological target named the dopamine type 2 receptor (DRD2)) scores.\n\n',
                }
        self.task2objective_mul = {
                '1': 'I want to maximize QED score, maximize JNK3 score, and minimize SA score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                '2': 'I want to maximize QED score, maximize GSK3$\beta$ score, and minimize SA score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                '3': 'I want to maximize QED score, maximize JNK3 score, minimize GSK3$\beta$ score, minimize SA score and minimize DRD2 score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                }
        self.requirements = """\n\nYour output should follow the format: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. Here are the requirements:\n
        \n\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your propsosed molecule.\n3. The molecule should be valid.
        """
        self.task=None

    def edit(self, mating_tuples, mutation_rate):
        task = self.task
        task_definition = self.task2description_mul[task[0]]
        task_objective = self.task2objective_mul[task[0]]

        parent = []
        parent.append(random.choice(mating_tuples))
        parent.append(random.choice(mating_tuples))
        parent_mol = [t[1] for t in parent]
        parent_scores = [t[0] for t in parent]
        try:
            mol_tuple = ''
            for i in range(2):
                tu = '\n[' + Chem.MolToSmiles(parent_mol[i]) + ',' + str(parent_scores[i]) + ']'
                mol_tuple = mol_tuple + tu
            prompt = task_definition + mol_tuple + task_objective + self.requirements
            _, r = query_LLM(prompt)
            proposed_smiles = re.search(r'\\box\{(.*?)\}', r).group(1)
            proposed_smiles = sanitize_smiles(proposed_smiles)
            print(proposed_smiles)
            assert proposed_smiles != None
            new_child = Chem.MolFromSmiles(proposed_smiles)

            return new_child
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            new_child = co.crossover(parent_mol[0], parent_mol[1])
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
            return new_child
    
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