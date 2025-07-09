import numpy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers
import crossover as co
import mutate as mu
import re
import random
MINIMUM = 1e-10
def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))
class Llama3:
    def __init__(self):
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        # self.model = AutoModelForCausalLM.from_pretrained(model_id,device_map = 'cuda')
        self.wild_type={
            'GB1': 'VDGV',
            'TrpB': 'VFVS',
            'Syn-3bfo': 'SKLQICVEPTSQKLMPGSTLVLQCVAVGSPIPHYQWFKNELPLTHETKKLYMVPYVDLEHQGTYWCHVYNDRDSQDSKKVEIIID',
            'AAV':'DEEEIRTTNPVATEQYGSVSTNLQRGNR',
            'GFP':'SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
        }
        self.task={
            'GB1': 'ability to bind affinity-based sequence enrichment',
            'TrpB': 'ability to couple growth to the rate of tryptophan formation',
            'Syn-3bfo': 'ability to perform its target function',
            'AAV':"ability to package DNA payload, i.e, for gene delivery. ",
            'GFP':'fluorescence properties as a biomarker',
        }
        self.dataset_description = {
            'GB1': \
'''We are focusing on changes to a limited subset of amino acids within the sequence. The provided subset protein sequences come from B1 domain of streptococcal protein G, with sequence:
```
MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
```
Each subset protein sequence represents specific amino acid substitutions at four key positions: 39, 40, 41, and 54, denoted using the single-letter amino acid code.
''',

            'TrpB': \
'''We are focusing on changes to a limited subset of amino acids within the sequence. The provided subset protein sequences come from beta-subunit of tryptophan synthase, with sequence:
```
MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMGKTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSVSAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIRLEHHHHHH
```
Each subset protein sequence represents specific amino acid substitutions at four key positions: 183, 184, 227, and 228, denoted using the single-letter amino acid code.
''',
            # 'Syn-3bfo': 'The provided subset protein sequences come from wild type protein: CRYSTAL STRUCTURE OF IG-LIKE C2-TYPE 2 DOMAIN of the human Mucosa-associated lymphoid tissue lymphoma translocation protein 1, with wild type sequence: SKLQICVEPTSQKLMPGSTLVLQCVAVGSPIPHYQWFKNELPLTHETKKLYMVPYVDLEHQGTYWCHVYNDRDSQDSKKVEIIID. You can do mutation on every amino acid without changing the length.\n',
            'Syn-3bfo': \
'''
The provided subset protein sequences come from wild type protein: crystal structure of Ig-like C2-type 2 domain of the human Mucosa-associated lymphoid tissue lymphoma translocation protein 1, with sequence:
```
SKLQICVEPTSQKLMPGSTLVLQCVAVGSPIPHYQWFKNELPLTHETKKLYMVPYVDLEHQGTYWCHVYNDRDSQDSKKVEIIID
```
You can do mutation on every amino acid without changing the length.
''',
#             'AAV': \
# '''
# The provided dataset comprises protein sequences derived from mutated variants of the Adeno-Associated Virus 2 (AAV2) Capsid protein VP1. These mutations are designed to study or improve the properties of AAV2, such as its infectivity, tissue tropism, stability, or immune evasion. The dataset focuses on exploring how sequence variations influence AAV2 functionality, particularly in the context of gene delivery and therapeutic applications. The wild-type VP1 protein sequence in this dataset is:
# ```
# MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL
# ```
# Each subset of protein sequences represents specific amino acid substitutions at 28 key positions, ranging from positions 561 to 588, using the single-letter amino acid code.
# '''
            'AAV': \
'''
We are focusing on changes to a limited subset of amino acids within the AAV2 Capsid protein VP1 sequence. The provided subset protein sequences are derived from the VP1 region with the wild-type sequence:
```
MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL
```

the part we can mutate or modify is "DEEEIRTTNPVATEQYGSVSTNLQRGNR" which is in the position from 561-588 from this wild-type whole protein sequence. These mutations are designed to study or enhance critical properties of AAV2, such as infectivity, tissue tropism, stability, or immune evasion. The dataset aims to explore how sequence variations in this specific region impact AAV2 functionality, particularly for applications in gene delivery and therapeutics.
''',
            'GFP': \
'''
The provided dataset contains protein sequences derived from mutated variants of fluorescent proteins, specifically originating from the Green Fluorescent Protein (GFP) domain. These mutations are engineered or naturally occurring modifications of GFP to explore or enhance its fluorescent properties, stability, or other functional characteristics. The wild-type protein sequence in this dataset is:
```
SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
```
You can do mutation on every amino acid without changing the length.
''',
# "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
            # S K L Q I C V E P T S Q K L M P G S T L V L Q C V A V G S P I P H Y Q W F K N E L P L T H E T K K L Y M V P Y V D L E H Q G T Y W C H V Y N D R D S Q D S K K V E I I I D
            # SKLQICVEPTSQKLMPGSTLVLQCVAVGSPIPHYQWFKNELPLTHETKKLYMVPYVDLEHQGTYWCHVYNDRDSQDSKKVEIIID
    }
        self.restrict = {
            'Syn-3bfo': "\nYou MUST propose sequence that is DIFFERENT from the wild type sequence: S K L Q I C V E P T S Q K L M P G S T L V L Q C V A V G S P I P H Y Q W F K N E L P L T H E T K K L Y M V P Y V D L E H Q G T Y W C H V Y N D R D S Q D S K K V E I I I D \n",
            'TrpB': "\nYou MUST propose sequence that is DIFFERENT from the wild type sequence: M K G Y F G P Y G G Q Y V P E I L M G A L E E L E A A Y E G I M K D E S F W K E F N D L L R D Y A G R P T P L Y F A R R L S E K Y G A R V Y L K R E D L L H T G A H K I N N A I G Q V L L A K L M G K T R I I A E T G A G Q H G V A T A T A A A L F G M E C V I Y M G E E D T I R Q K L N V E R M K L L G A K V V P V K S G S R T L K D A I D E A L R D W I T N L Q T T Y Y V F G S V V G P H P Y P I I V R N F Q K V I G E E T K K Q I P E K E G R L P D Y I V A C V S G G S N A A G I F Y P F I D S G V K L I G V E A G G E G L E T G K H A A S L L K G K I G Y L H G S K T F V L Q D D W G Q V Q V S H S V S A G L D Y S G V G P E H A Y W R E T G K V L Y D A V T D E E A L D A F I E L S R L E G I I P A L E S S H A L A Y L K K I N I K G K V V V V N L S G R G D K D L E S V L N H P Y V R E R I R L E H H H H H H \n",
            
        }
        self.pipeline = transformers.pipeline(
            "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
        # self.api=LLMCaller('llama3.1-8b-api', 0.5)
    def edit(self, mating_tuples, mutation_rate, dataset='GB1', K=-1, BK=-1, mean=0, std=0):
        dataset_description = self.dataset_description[dataset]
        constrain = ''
        # print(mating_tuples)
        try:
            parent = []
            parent.append(random.choice(mating_tuples))
            parent.append(random.choice(mating_tuples))
            parent_mol = [t[1] for t in parent]
            parent_scores = [t[0] for t in parent]
            for retry in range(5):

                if K != -1:
                    constrain = f'* The proposed sequence must have a hamming distance between 1 and {K} from the {" ".join(self.wild_type[dataset])}.'
                if BK != -1:
                    constrain = f'* The proposed sequence must have a hamming distance between 1 and {BK} from the {" ".join(parent_mol[0])} and {" ".join(parent_mol[1])}.'
                # mol_tuple = f'Here are the protein sequences with {len(parent_mol[0])} amino acids, and their fitness scores:'  
                # for i in range(2):
                #     tu = f'\n{i+1}. Protein sequence: ' + ' '.join(parent_mol[i]) + ', Fitness score: ' + str(parent_scores[i])
                #     mol_tuple = mol_tuple + tu
                # #TODO add special token
                content = \
'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a world-class assistant specializing in protein engineering, fitness optimization, and sequence design. Your expertise lies in analyzing sequence-function relationships, interpreting experimental data, and proposing rational modifications to optimize protein fitness.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

You will carry out a multi-round directed evolution experiment with the following protein sequence, aimed at improving protein's {task} via protein fitness optimization.

### Protein fitness optimization
The fitness score reflects the efficacy or functionality for a desired application, from chemical synthesis to bioremediation and therapeutics. Protein fitness optimization can be thought of as navigating a protein fitness landscape, a mapping of amino acid sequences to fitness values, to find higher-fitness variants. Specifically, it is achieved by making crossover and mutations on the given sequences.

{dataset_description}

### Parent protein sequences
Here are the parent protein sequences that you will be modifying from. Each sequence comes with {seq_len} amino acids and its fitness score is also provided.

Protein sequence 1 (fitness score: {score1})
```
{protein_seq_1}
```

Protein sequence 2 (fitness score: {score2})
```
{protein_seq_2}
```

### Instructions
Follow the instructions below to propose a new protein:
* Your proposal should focus on maximizing fitness while considering structural and functional plausibility.
* You can propose it via making crossover or mutation on the parent sequences.
* You can also propose a new sequence based on your knowledge.
* Your proposed sequence MUST have the same length as the parent sequences.
* DO NOT propose sequence that is identical with the parent or the wild type sequences. 
* Your output MUST ONLY include: \\box{{Protein}}. 
* The current pool has mean fitness as {mean}, standard deviation as {std}.
* DO NOT explain.
{constrain}

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>'''

                prompt = content.format(task=self.task[dataset], dataset_description=dataset_description, seq_len=len(parent_mol[0]), 
                                        protein_seq_1 = ' '.join(parent_mol[0]), protein_seq_2 = ' '.join(parent_mol[1]), score1=parent_scores[0],score2=parent_scores[1],
                                        constrain = constrain, mean=mean,std=std)

                try:
                    # if dataset.__contains__('Syn'):
                    if dataset =='AAV' or dataset == 'GFP':
                        if dataset == 'GFP':
                            max_new_tokens = 256
                        elif dataset == 'AAV':
                            max_new_tokens = 64
                        outputs = self.pipeline(prompt, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, temperature=0.5)[0]['generated_text'].removeprefix(prompt)
                        # outputs = self.api.get_response(prompt)
                        # print("here")
                        # print(self.pipeline(prompt, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, temperature=0.5)[0]['generated_text'])
                        # exit()
                    elif dataset.__contains__('Syn'):
                        outputs = self.pipeline(prompt, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id, temperature=0.5)[0]['generated_text'].removeprefix(prompt)
                        # outputs = self.api.get_response(prompt)
                        # print(outputs)
                        # exit(0)
                    else:
                        # print(prompt)
                        # exit()
                        outputs = self.pipeline(prompt, pad_token_id=self.tokenizer.eos_token_id)[0]['generated_text'].removeprefix(prompt)
                    # print('Retry ', retry , outputs)

                    proposed_smiles = re.search(r'\\box\{(.*?)\}', outputs).group(1).replace(' ', '')
                    # proposed_smiles = sanitize_smiles(proposed_smiles)
                    # print(proposed_smiles)
                    # exit()
                    # print(len(proposed_smiles), len(parent_mol[0]))
                    assert proposed_smiles != None, "Invalid"
                    assert len(proposed_smiles) == len(parent_mol[0]), "Uneven length"
                    assert proposed_smiles != parent_mol[0] and proposed_smiles != parent_mol[1], "Repeat"
                    assert set(proposed_smiles).issubset(set('ARNDCQEGHILKMFPSTWYV')), "Illegal protein"
                    # print(proposed_smiles)
                    new_child = proposed_smiles
                    break
                except Exception as e:
                    # print(f"{type(e).__name__} {e}")
                    continue
            return new_child, parent_mol[0],  parent_mol[1]
        except Exception as e:
            # print(f"{type(e).__name__} {e}")
            new_child = co.crossover_seq(parent_mol[0], parent_mol[1])
            if new_child is not None:
                new_child = mu.mutate_seq(new_child, mutation_rate)
            return new_child, parent_mol[0],  parent_mol[1]
    
if __name__ == "__main__":
    model = Llama3()
    # model=model.to('cuda')
    print(model.edit([[0.001791394,"KWNA"],[0.007212719,"ARAF"],[2.31199298,"MRFG"],[0.022350843,'LDVA']],0.0))
