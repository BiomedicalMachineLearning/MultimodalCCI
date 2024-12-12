import pandas as pd
import numpy as np
import anndata as ad
from typing import Dict, List, Optional, Union
from copy import deepcopy

class CCIData:
    """
    Class to store and manage Cell-Cell Interaction (CCI) data
    
    Attributes:
        metadata (Dict): Metadata for sample
        n_spots (int): Number of spots in the sample
        cci_scores (Dict): CCI score dataframe for each LR pair
        p_values (Dict): P-values dataframe for each LR pair
        adata (AnnData): AnnData object
        networks (Dict): Calculated CCI networks
    """
    
    
    def __init__(self, 
                 metadata: Dict = {}, 
                 n_spots: int = None, 
                 cci_scores: Dict = {}, 
                 p_values: Dict = {}, 
                 adata: ad.AnnData = None
                 ):
        self.metadata = metadata
        self.n_spots = n_spots
        self.cci_scores = cci_scores
        self.p_values = p_values
        self.adata = adata
        self.networks = {}
        
          
    def get_sample_metadata(self, sample_id: str) -> Dict:
        """
        Get metadata for a sample
        
        Args:
            sample_id: Sample ID
        
        Returns:
            Metadata for the sample
        """
        return self.metadata.get(sample_id)
    
    
    def get_sample_n_spots(self, sample_id: str) -> Optional[int]:
        """
        Get number of spots for a sample
        
        Args:
            sample_id: Sample ID
        
        Returns:
            Number of spots for the sample
        """
        return self.n_spots.get(sample_id)
    
    
    def get_sample_cci_scores(self, sample_id: str) -> Optional[pd.DataFrame]:
        """
        Get CCI scores for a sample
        
        Args:
            sample_id: Sample ID
        
        Returns:
            CCI scores for the sample
        """
        return self.cci_scores.get(sample_id)
    
    
    def get_sample_p_values(self, sample_id: str) -> Optional[pd.DataFrame]:
        """
        Get p-values for a sample
        
        Args:
            sample_id: Sample ID
        
        Returns:
            P-values for the sample
        """
        return self.p_values.get(sample_id)
    
    
    def get_adata(self) -> Optional[ad.AnnData]:
        """
        Get AnnData object
        
        Returns:
            AnnData object
        """
        return self.adata
    
    
    def copy(self) -> 'CCIData':
        """
        Create a copy of the CCIData object
        
        Returns:
            Copy of the CCIData object
        """
        cci_data = CCIData()
        cci_data.metadata = self.metadata.deep_copy()
        cci_data.n_spots = self.n_spots.copy()
        cci_data.cci_scores = self.cci_scores.deep_copy()
        cci_data.p_values = self.p_values.deep_copy()
        cci_data.adata = self.adata.deep_copy()
        
        return cci_data
    
    
    def rename_celltypes(self, replacements: Dict[str, str]) -> 'CCIData':
        """Renames cell types in a CCIData.

        Args:
            replacements (dict): A dictionary of replacements, where the keys are the 
                old cell type names and the values are the new cell type names.

        Returns:
            dict: A dict of LR matrices with the cell type names replaced.
        """

        renamed_cci_data = self.copy()
        
        for lr_pair in renamed_cci_data.cci_scores.keys():
            renamed_cci_data.cci_scores[lr_pair].rename(
                index=replacements,
                columns=replacements,
                inplace=True)
            
        for lr_pair in renamed_cci_data.p_values.keys():
            renamed_cci_data.p_values[lr_pair].rename(
                index=replacements,
                columns=replacements,
                inplace=True)

        return renamed_cci_data
    