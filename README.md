
## Prerequisites
Before you can run this project, you'll need the following installed:
- Python 3.11.5
- pip 23.2.1
- Visual Studio Code
- Java version: 21.0.3

## Setting Up the Environment

### Python Environment Setup
To get started, set up a Python virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

Next, install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the safetesnors generator 

```bash
cd python 
python -m json_to_bin

```

### Setting Up Java in Visual Studio Code
Install the "Language Support for Java(TM) by Red Hat" extension in VS Code. You can do this from the Visual Studio Code Marketplace or directly within VS Code by searching for the extension in the Extensions view (`Ctrl+Shift+X`).

## Running the Project
``` cd /demo/src/main/java/com/example/ ```

### Running Java Programs in Interactive Mode
To run Java programs in interactive mode, ensure that you've correctly configured Java in VS Code. Open your Java file, then start the interactive mode by pressing `F5` or selecting `Run > Start Debugging` from the menu.

### Verifying and Testing
Use the following commands to verify and test your project components:
  ```
cd python
  ```

- **Verification**:
  ```bash
  python -m test
  ```

- **Comparing JSON Files**:
  ```bash
  python compare_jsonl.py <vectors_file.jsonl> <contents_file.jsonl>
  ```

Replace `<vectors_file.jsonl>` and `<contents_file.jsonl>` with the paths to your JSONL files as required.

