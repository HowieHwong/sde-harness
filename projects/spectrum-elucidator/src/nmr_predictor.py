"""
NMR prediction module for the Spectrum Elucidator Toolkit.

This module provides functionality to predict NMR spectra for generated SMILES
structures using web scraping from NMRDB and LLM fallback prediction.
"""

# Core dependencies
import time
import pandas as pd
import openai
from bs4 import BeautifulSoup
import re
import urllib.parse
from typing import List, Tuple, Optional, Dict, Any
import logging
from .similarity import compare_c_nmr_strings, parse_c_nmr

# Optional dependencies - import with fallbacks
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available. Web scraping will be disabled.")

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Molecular validation will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Some calculations may be limited.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Plotting will be disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Image processing will be disabled.")

# Note: easyocr is intentionally excluded due to heavy PyTorch dependencies
# We use BeautifulSoup text extraction instead, which is more reliable for HTML content


class NMRPredictor:
    """Predict NMR spectra for SMILES structures using web scraping and LLM fallback."""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 headless: bool = True,
                 timeout: int = 10,
                 use_web_scraping: bool = True):
        """
        Initialize the NMR predictor.
        
        Args:
            openai_api_key: OpenAI API key for LLM fallback prediction
            headless: Whether to run browser in headless mode
            timeout: Timeout for web scraping operations
            use_web_scraping: Whether to enable web scraping (requires Selenium)
        """
        self.openai_api_key = openai_api_key
        self.headless = headless
        self.timeout = timeout
        self.use_web_scraping = use_web_scraping and SELENIUM_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        if not SELENIUM_AVAILABLE and use_web_scraping:
            self.logger.warning("Selenium not available. Web scraping disabled.")
            self.use_web_scraping = False
    
    def collect_nmr_records(self, text: str) -> List[str]:
        """
        Extract NMR records from text using regex pattern.
        
        Args:
            text: Text containing NMR data
            
        Returns:
            List of extracted NMR records
        """
        # Define regex pattern to capture NMR records
        nmr_pattern = r'NMR:\s*(.*?)(?=\.\s| loading|\n|$)'
        
        # Use re.findall to extract all matches
        nmr_records = re.findall(nmr_pattern, text, re.DOTALL)
        
        # Clean up records
        nmr_records = [record.strip() for record in nmr_records]
        
        return nmr_records
    
    def get_molecular_formula(self, smiles: str) -> str:
        """
        Get molecular formula from SMILES using RDKit.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Molecular formula string
        """
        if not RDKIT_AVAILABLE:
            return "RDKit not available"
        
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                return "Invalid SMILES string"
            
            molecular_formula = Chem.rdMolDescriptors.CalcMolFormula(molecule)
            return molecular_formula
        except Exception as e:
            self.logger.error(f"Error calculating molecular formula: {e}")
            return "Error"
    
    def capture_text_from_webpage(self, url: str) -> str:
        """
        Capture text from webpage using Selenium and BeautifulSoup.
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted text content
        """
        if not self.use_web_scraping:
            self.logger.warning("Web scraping is disabled")
            return ""
        
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)
            
            # Navigate to URL
            driver.get(url)
            
            # Wait for page to load
            time.sleep(6)
            
            # Try to click "I agree" button if present
            try:
                agree_button = WebDriverWait(driver, 7).until(
                    EC.element_to_be_clickable((By.XPATH, '//button[text()="I agree"]'))
                )
                agree_button.click()
                time.sleep(2)
            except Exception as e:
                self.logger.debug(f"No agree button found: {e}")
            
            # Get page source
            page_source = driver.page_source
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            extracted_text = soup.get_text(separator=' ')
            
            return extracted_text
            
        except Exception as e:
            self.logger.error(f"Error scraping webpage {url}: {e}")
            return ""
        finally:
            if driver:
                driver.quit()
    
    def get_nmr_from_web(self, smiles: str) -> Tuple[List[str], List[str]]:
        """
        Get NMR data from NMRDB website.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Tuple of (C_NMR_records, H_NMR_records)
        """
        if not self.use_web_scraping:
            self.logger.info("Web scraping disabled, returning empty results")
            return [], []
        
        try:
            # Encode SMILES for URL
            encoded_smiles = urllib.parse.quote(smiles)
            
            # URLs for NMR prediction
            url_h = f"https://www.nmrdb.org/service.php?name=nmr-1h-prediction&smiles={encoded_smiles}"
            url_c = f"https://www.nmrdb.org/service.php?name=nmr-13c-prediction&smiles={encoded_smiles}"
            
            # Scrape H-NMR
            h_nmr_text = self.capture_text_from_webpage(url_h)
            h_nmr_records = self.collect_nmr_records(h_nmr_text)
            
            # Scrape C-NMR
            c_nmr_text = self.capture_text_from_webpage(url_c)
            c_nmr_records = self.collect_nmr_records(c_nmr_text)
            
            self.logger.info(f"Web scraping successful for {smiles}: H={len(h_nmr_records)}, C={len(c_nmr_records)}")
            
            return c_nmr_records, h_nmr_records
            
        except Exception as e:
            self.logger.error(f"Error getting NMR from web for {smiles}: {e}")
            return [], []
    
    def predict_nmr_with_llm(self, smiles: str, nmr_type: str = "both") -> Tuple[List[str], List[str]]:
        """
        Predict NMR using LLM when web scraping fails.
        
        Args:
            smiles: SMILES representation of the molecule
            nmr_type: Type of NMR to predict ("H", "C", or "both")
            
        Returns:
            Tuple of (C_NMR_records, H_NMR_records)
        """
        if not self.openai_api_key:
            self.logger.warning("No OpenAI API key provided for LLM prediction")
            return [], []
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Create prompt for NMR prediction
            prompt = f"""You are an expert chemist specializing in NMR spectroscopy. 
Given the SMILES structure: {smiles}

Please predict the {nmr_type} NMR spectrum. Consider:
1. Chemical environment of each atom
2. Typical chemical shift ranges
3. Multiplicity patterns
4. Integration values

Provide your response in this exact format:
H-NMR: [chemical_shift] ([integration], [multiplicity], J = [coupling] Hz), [more peaks...]
C-NMR: [chemical_shift], [more peaks...]

Example format:
H-NMR: 0.87 (3H, t, J = 6.5 Hz), 1.30 (2H, m), 7.10-7.32 (5H, m)
C-NMR: 14.0, 22.6, 32.8, 127.8, 128.4, 140.6

Please provide the NMR prediction:"""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert chemist specializing in NMR spectroscopy."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            
            # Extract NMR records from response
            h_nmr_records = self.collect_nmr_records(response_text)
            c_nmr_records = self.collect_nmr_records(response_text)
            
            self.logger.info(f"LLM prediction successful for {smiles}: H={len(h_nmr_records)}, C={len(c_nmr_records)}")
            
            return c_nmr_records, h_nmr_records
            
        except Exception as e:
            self.logger.error(f"Error predicting NMR with LLM for {smiles}: {e}")
            return [], []
    
    def get_nmr_prediction(self, smiles: str, fallback_to_llm: bool = True) -> Tuple[List[str], List[str]]:
        """
        Get NMR prediction for a SMILES structure.
        
        Args:
            smiles: SMILES representation of the molecule
            fallback_to_llm: Whether to use LLM if web scraping fails
            
        Returns:
            Tuple of (C_NMR_records, H_NMR_records)
        """
        # Validate SMILES
        if not self._validate_smiles(smiles):
            self.logger.warning(f"Invalid SMILES: {smiles}")
            return [], []
        
        # Try web scraping first if enabled
        c_nmr_records, h_nmr_records = [], []
        if self.use_web_scraping:
            c_nmr_records, h_nmr_records = self.get_nmr_from_web(smiles)
        
        # If web scraping failed and LLM fallback is enabled
        if fallback_to_llm and (not c_nmr_records or not h_nmr_records):
            self.logger.info(f"Web scraping failed for {smiles}, trying LLM prediction")
            c_nmr_records, h_nmr_records = self.predict_nmr_with_llm(smiles)
        
        return c_nmr_records, h_nmr_records
    
    def _validate_smiles(self, smiles: str) -> bool:
        """
        Basic SMILES validation.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if SMILES appears valid
        """
        if not smiles:
            return False
        
        if not RDKIT_AVAILABLE:
            # Basic validation without RDKit
            return bool(re.match(r'^[A-Za-z0-9()\[\]=#@+-]+$', smiles))
        
        try:
            molecule = Chem.MolFromSmiles(smiles)
            return molecule is not None
        except:
            return False
    
    def format_nmr_for_comparison(self, nmr_records: List[str], nmr_type: str = "C") -> str:
        """
        Format NMR records for similarity comparison.
        
        Args:
            nmr_records: List of NMR record strings
            nmr_type: Type of NMR ("H" or "C")
            
        Returns:
            Formatted NMR string for comparison
        """
        if not nmr_records:
            return ""
        
        # Join records with proper formatting
        if nmr_type == "C":
            # For C-NMR, typically just chemical shifts
            formatted = "δ " + ", ".join(nmr_records)
        else:
            # For H-NMR, keep full format
            formatted = " ".join(nmr_records)
        
        return formatted
    
    def calculate_nmr_similarity(self, 
                                target_nmr: str, 
                                predicted_nmr: str, 
                                nmr_type: str = "C",
                                tolerance: float = 0.20) -> float:
        """
        Calculate similarity between target and predicted NMR spectra.
        
        Args:
            target_nmr: Target NMR spectrum string
            predicted_nmr: Predicted NMR spectrum string
            nmr_type: Type of NMR ("H" or "C")
            tolerance: Tolerance for peak matching in ppm
            
        Returns:
            Similarity score between 0 and 1
        """
        if not target_nmr or not predicted_nmr:
            return 0.0
        
        try:
            if nmr_type == "C":
                # Use the similarity.py comparison for C-NMR
                result = compare_c_nmr_strings(
                    target_nmr, 
                    predicted_nmr, 
                    tol_ppm=tolerance
                )
                
                # Extract F1 score as similarity
                similarity = result["metrics"]["f1"]
                return similarity
            else:
                # For H-NMR, use a simpler approach
                # Parse peaks and calculate basic similarity
                target_peaks = self._parse_h_nmr_peaks(target_nmr)
                predicted_peaks = self._parse_h_nmr_peaks(predicted_nmr)
                
                if not target_peaks or not predicted_peaks:
                    return 0.0
                
                # Calculate similarity based on peak matching
                matched_peaks = 0
                total_peaks = max(len(target_peaks), len(predicted_peaks))
                
                for target_peak in target_peaks:
                    for predicted_peak in predicted_peaks:
                        if abs(target_peak - predicted_peak) <= tolerance:
                            matched_peaks += 1
                            break
                
                return matched_peaks / total_peaks if total_peaks > 0 else 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating NMR similarity: {e}")
            return 0.0
    
    def _parse_h_nmr_peaks(self, nmr_string: str) -> List[float]:
        """
        Parse H-NMR peaks to extract chemical shifts.
        
        Args:
            nmr_string: H-NMR spectrum string
            
        Returns:
            List of chemical shift values
        """
        try:
            # Use the parse_c_nmr function from similarity.py as reference
            # but adapt for H-NMR patterns
            peaks = parse_c_nmr(nmr_string, keep_ranges_as_center=True)
            return peaks
        except Exception as e:
            self.logger.error(f"Error parsing H-NMR peaks: {e}")
            return []


def test_nmr_predictor():
    """Test function for the NMR predictor."""
    # Test SMILES
    test_smiles = "CCCCC1=CC=CC=C1"  # Pentylbenzene
    
    # Initialize predictor (without API key for testing)
    predictor = NMRPredictor(use_web_scraping=False)  # Disable web scraping for testing
    
    print(f"Testing NMR prediction for: {test_smiles}")
    print(f"Molecular formula: {predictor.get_molecular_formula(test_smiles)}")
    
    # Test web scraping (disabled)
    print("\nTesting web scraping...")
    if predictor.use_web_scraping:
        c_nmr, h_nmr = predictor.get_nmr_from_web(test_smiles)
        print(f"C-NMR records: {c_nmr}")
        print(f"H-NMR records: {h_nmr}")
    else:
        print("Web scraping disabled for testing")
    
    # Test formatting
    test_c_nmr = ["170.2", "151.3", "128.1", "77.16", "55.4"]
    test_h_nmr = ["0.87 (3H, t)", "1.30 (2H, m)", "7.10-7.32 (5H, m)"]
    
    formatted_c = predictor.format_nmr_for_comparison(test_c_nmr, "C")
    formatted_h = predictor.format_nmr_for_comparison(test_h_nmr, "H")
    
    print(f"Formatted C-NMR: {formatted_c}")
    print(f"Formatted H-NMR: {formatted_h}")
    
    # Test similarity calculation
    target_c_nmr = "δ 170.2, 151.3, 128.1, 77.16, 55.4"
    predicted_c_nmr = "δ 170.21, 151.1, 128.3, 77.0, 29.8"
    
    similarity = predictor.calculate_nmr_similarity(
        target_c_nmr, predicted_c_nmr, "C", tolerance=0.20
    )
    print(f"C-NMR similarity: {similarity:.3f}")


if __name__ == "__main__":
    test_nmr_predictor()
