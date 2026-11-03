import pandas as pd

def postprocessing(
        scalability_results:pd.DataFrame
) -> pd.DataFrame:
    """Postprocesses the results dataframe.
    
    Args:
        scalability_results: Results data frame testing scalability.

    Returns:
        A post-processed data frame ready for plotting and analysis.
    """