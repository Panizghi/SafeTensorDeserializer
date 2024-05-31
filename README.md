
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
```
cd demo 
mvn clean install 
mvn clean package 
mvn exec:java -Dexec.mainClass="com.example.SafeTensorsDeserializer"
```

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

