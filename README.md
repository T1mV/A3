# Assignment 3 - Multi-Agent Hospital Simulation

## Running the code

### 1. Creating a virtual environment and installing dependencies
First we need to create a venv. For this we can use: 
```bash 
python -m venv venv
```

To activate you can use in Windows powershell
```bash
.\venv\Scripts\Activate.ps1
```
And for Linux:
```bash
source venv/bin/activate
```

After this we can install the dependencies:
```bash
python -m pip install -r requirements.txt
```

## 2. Training the ML models (optional when the model pkl files are already included)
```bash
python 0_train_models.py
```   

## 3. Running the baseline simulation (optional since this is the same as given and is the baseline method)
```bash
python 1_basic_mcp.py
```

## 4. Running the LLM simulation
The model used is the GPT-OSS:20B through the Ollama API. Before running the file it is necessary that you get an API key from its site. After collecting this API key, you can set it up in Windows PowerShell with:

```bash
$env:OLLAMA_API_KEY="your_api_key"
```
In Linux, you can use:

```bash
export OLLAMA_API_KEY="your_api_key"
```

After this, you can run the simulation python file:

```bash 
python 2_llm_mcp.py
```

The Pandas Dataframe are saved in the results directory as a .pkl

## 5. Creating the figures

The figure part of the assignment was done in a separate file. You can run it with the following command:

```bash
python 3_analyse_results.py
```