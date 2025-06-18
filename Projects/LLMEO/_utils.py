from typing import Dict, List, Optional
import pandas as pd
import re
def make_text_for_existing_tmcs(
    df: pd.DataFrame, 
    lig_charge: Dict[str, int], 
    props: List[str]
) -> str:
    """
    Create formatted text representation of TMCs with their properties.
    
    Args:
        df: DataFrame containing TMC data
        lig_charge: Dictionary mapping ligands to their charges
        props: List of property names to include
        
    Returns:
        Formatted string containing TMC information
    """
    lines = []
    for _, row in df.iterrows():
        # Construct TMC string
        tmc = "Pd_" + "_".join([row["lig1"], row["lig2"], row["lig3"], row["lig4"]])
        
        # Calculate total charge
        total_charge = 2 + sum(lig_charge[row[f"lig{i}"]] for i in range(1, 5))
        
        # Get property values
        prop_values = [str(round(row[prop], 3)) for prop in props]
        
        # Format line
        line = "{" + ", ".join([tmc, str(total_charge)] + prop_values) + "}"
        lines.append(line)
        
    return "\n".join(lines)

def retrive_tmc_from_message(message: str, expected_returns: int = 1) -> List[str]:
    """
    Extract TMC strings from LLM response message.
    
    Args:
        message: Response message from LLM
        expected_returns: Expected number of TMCs to extract
        
    Returns:
        List of extracted TMC strings
    """
    # TMC pattern matching regex
    pattern = r"Pd_(\w{6})-subgraph-(\d+)_(\w{6})-subgraph-(\d+)_(\w{6})-subgraph-(\d+)_(\w{6})-subgraph-(\d+)"
    
    # Possible delimiters for TMC in message
    delimiters = [
        "*TMC*", "<<<TMC>>>:", "<TMC>", "TMC:", " TMC"
    ]
    
    # Try to split message using different delimiters
    message_parts = None
    for delimiter in delimiters:
        if delimiter in message:
            message_parts = message.split(delimiter)
            break
            
    if message_parts is None:
        print("Unidentified pattern for splitting the LLM message.")
        return []
    
    # Extract TMCs
    tmcs = []
    for i in range(expected_returns):
        try:
            idx = -expected_returns + i
            match = re.search(pattern, message_parts[idx])
            
            if match:
                tmc = match.group()
                if len(tmc.split("_")) == 5:  # Validate TMC format
                    tmcs.append(tmc)
                else:
                    print(f"Invalid TMC format: {tmc}")
                    
        except IndexError:
            continue
    
    return tmcs

def find_tmc_in_space(df: pd.DataFrame, tmcs: List[str]) -> Optional[pd.DataFrame]:
    """
    Find TMCs in the search space by checking all possible rotations of ligands.
    
    Args:
        df: DataFrame containing the TMC search space
        tmcs: List of TMC strings to search for
        
    Returns:
        DataFrame containing matched TMCs, or None if no matches found
    """
    matched_tmcs = []
    
    for tmc in tmcs:
        if tmc is None:
            continue
            
        # Get ligands from TMC string
        ligs = tmc.split("_")[1:]
        
        # Check all possible rotational combinations of ligands
        rotations = [
            ligs[i:] + ligs[:i] for i in range(4)
        ]
        
        # Search for each rotation in the DataFrame
        for rot_ligs in rotations:
            match_df = df[
                (df["lig1"] == rot_ligs[0]) &
                (df["lig2"] == rot_ligs[1]) &
                (df["lig3"] == rot_ligs[2]) &
                (df["lig4"] == rot_ligs[3])
            ]
            
            if len(match_df):
                matched_tmcs.append(match_df)
                # print("matched_tmcs: "+str(matched_tmcs))
                break  # Found a match, move to next TMC
                
    return pd.concat(matched_tmcs) if matched_tmcs else None
