# LLMEO Project Setup

## Original Code Repository
[https://github.com/deepprinciple/llmeo](https://github.com/deepprinciple/llmeo)

## Installation Instructions

1. Navigate to project folder:
   ```bash
   cd Projects/LLMEO
   ```


2. Add your API key as environment variable
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```

3. Download required dataset:
    ```
    wget https://zenodo.org/records/14328055/files/ground_truth_fitness_values.csv -P data/
    ```

4. Set up conda environment:
    ```
    conda env create -f environment.yml
    conda activate ScienceBench_LLMEO
    ```

4.5 (Optional)
    If you are inconfident with you setup, you can run the test file:
    ```
    python test.py
    ```
    

5. Run the application:
    ```
    python main.py
    ```
    
    